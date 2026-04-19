/*******************************************************
 Hula Valley Land Cover Classification (GEE)
 Complete corrected version with:
 - AOI geometry fix
 - Dynamic World mode labels
 - Working legend
 - RF + SVM classification
 - Exports
*******************************************************/

// -------------------------
// 0) AOI FIX
// -------------------------
var aoiGeom = ee.FeatureCollection(aoi).geometry();

print('AOI object:', aoi);
print('AOI geometry:', aoiGeom);

Map.centerObject(aoiGeom, 11);
Map.addLayer(aoiGeom, {color: 'red'}, 'AOI geometry');

// -------------------------
// 1) DATE RANGE + CLOUD MASK
// -------------------------
var start = '2024-04-01';
var end   = '2024-10-31';

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoiGeom)
  .filterDate(start, end)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30));

function maskS2(img) {
  var scl = img.select('SCL');
  var mask = scl.neq(3)   // cloud shadow
    .and(scl.neq(8))      // cloud medium probability
    .and(scl.neq(9))      // cloud high probability
    .and(scl.neq(10))     // thin cirrus
    .and(scl.neq(11));    // snow/ice
  return img.updateMask(mask).divide(10000);
}

s2 = s2.map(maskS2);

// -------------------------
// 2) FEATURE ENGINEERING
// -------------------------
function addIndices(img) {
  var ndvi  = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndwi  = img.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var mndwi = img.normalizedDifference(['B3', 'B11']).rename('MNDWI');
  var ndbi  = img.normalizedDifference(['B11', 'B8']).rename('NDBI');
  return img.addBands([ndvi, ndwi, mndwi, ndbi]);
}

var composite = s2.map(addIndices).median().clip(aoiGeom);

var bands = [
  'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
  'NDVI', 'NDWI', 'MNDWI', 'NDBI'
];

var featureImg = composite.select(bands);

Map.addLayer(
  featureImg,
  {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3},
  'Sentinel-2 RGB'
);

// -------------------------
// 3) DYNAMIC WORLD LABELS
// -------------------------
var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(aoiGeom)
  .filterDate(start, end)
  .select('label');

print('Dynamic World collection size:', dw.size());
print('First Dynamic World image:', dw.first());

var label = dw.mode()
  .rename('label')
  .clip(aoiGeom)
  .toByte();

print('Label band names:', label.bandNames());

var dwVis = {
  min: 0,
  max: 8,
  palette: [
    '419BDF', // 0 Water
    '397D49', // 1 Trees
    '88B053', // 2 Grass
    '7A87C6', // 3 Flooded vegetation
    'E49635', // 4 Crops
    'DFC35A', // 5 Shrub & scrub
    'C4281B', // 6 Built
    'A59B8F', // 7 Bare
    'B39FE1'  // 8 Snow & ice
  ]
};

Map.addLayer(label, dwVis, 'Dynamic World mode');

// Debug class layers
Map.addLayer(label.eq(0).selfMask(), {palette: ['0000FF']}, 'Water only');
Map.addLayer(label.eq(3).selfMask(), {palette: ['00FFFF']}, 'Flooded vegetation only');
Map.addLayer(label.eq(6).selfMask(), {palette: ['FF0000']}, 'Built only');

// Check class counts
var classCounts = label.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: aoiGeom,
  scale: 10,
  maxPixels: 1e9
});
print('Class counts:', classCounts);

// -------------------------
// 4) RESET UI + LEGEND
// -------------------------
// -------------------------
// 4) LEGEND
// -------------------------
var classNames = [
  'Water',
  'Trees',
  'Grass',
  'Flooded vegetation',
  'Crops',
  'Shrub & scrub',
  'Built',
  'Bare',
  'Snow & ice'
];

var classColors = [
  '419BDF',
  '397D49',
  '88B053',
  '7A87C6',
  'E49635',
  'DFC35A',
  'C4281B',
  'A59B8F',
  'B39FE1'
];

var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 12px',
    backgroundColor: 'white',
    border: '1px solid black'
  }
});

legend.add(ui.Label({
  value: 'Dynamic World Classes',
  style: {
    fontWeight: 'bold',
    fontSize: '14px',
    margin: '0 0 6px 0'
  }
}));

for (var i = 0; i < classNames.length; i++) {
  var colorBox = ui.Label('', {
    backgroundColor: '#' + classColors[i],
    padding: '8px',
    margin: '0 0 4px 0'
  });

  var description = ui.Label(i + ': ' + classNames[i], {
    margin: '0 0 4px 6px'
  });

  var row = ui.Panel(
    [colorBox, description],
    ui.Panel.Layout.Flow('horizontal')
  );

  legend.add(row);
}

// Add legend directly to the map
Map.add(legend);

// Re-add layers after ui.root.clear()
Map.centerObject(aoiGeom, 11);
Map.addLayer(aoiGeom, {color: 'red'}, 'AOI geometry');
Map.addLayer(
  featureImg,
  {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3},
  'Sentinel-2 RGB'
);
Map.addLayer(label, dwVis, 'Dynamic World mode');
Map.addLayer(label.eq(0).selfMask(), {palette: ['0000FF']}, 'Water only');
Map.addLayer(label.eq(3).selfMask(), {palette: ['00FFFF']}, 'Flooded vegetation only');
Map.addLayer(label.eq(6).selfMask(), {palette: ['FF0000']}, 'Built only');

// -------------------------
// 5) SAMPLE TRAINING DATA
// -------------------------
var samples = featureImg.addBands(label).stratifiedSample({
  numPoints: 1500,
  classBand: 'label',
  region: aoiGeom,
  scale: 10,
  seed: 42,
  geometries: false
});

var withRand = samples.randomColumn('rand', 42);
var train = withRand.filter(ee.Filter.lt('rand', 0.7));
var test  = withRand.filter(ee.Filter.gte('rand', 0.7));

print('Train size:', train.size());
print('Test size:', test.size());

// -------------------------
// 6) RANDOM FOREST
// -------------------------
var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: 200,
  bagFraction: 0.7,
  seed: 42
}).train({
  features: train,
  classProperty: 'label',
  inputProperties: bands
});

var rfClass = featureImg.classify(rf).rename('rf_class').toByte();
Map.addLayer(rfClass, dwVis, 'RF classification');

var rfTest = test.classify(rf);
var rfCM = rfTest.errorMatrix('label', 'classification');
print('RF confusion matrix:', rfCM);
print('RF overall accuracy:', rfCM.accuracy());
print('RF kappa:', rfCM.kappa());
print('RF producers accuracy:', rfCM.producersAccuracy());
print('RF users accuracy:', rfCM.consumersAccuracy());

// -------------------------
// 7) SVM
// -------------------------
var svm = ee.Classifier.libsvm({
  kernelType: 'RBF',
  gamma: 0.5,
  cost: 10
}).train({
  features: train,
  classProperty: 'label',
  inputProperties: bands
});

var svmClass = featureImg.classify(svm).rename('svm_class').toByte();
Map.addLayer(svmClass, dwVis, 'SVM classification');

var svmTest = test.classify(svm);
var svmCM = svmTest.errorMatrix('label', 'classification');
print('SVM confusion matrix:', svmCM);
print('SVM overall accuracy:', svmCM.accuracy());
print('SVM kappa:', svmCM.kappa());
print('SVM producers accuracy:', svmCM.producersAccuracy());
print('SVM users accuracy:', svmCM.consumersAccuracy());

// -------------------------
// 8) EXPORTS
// -------------------------
Export.table.toDrive({
  collection: withRand.select(bands.concat(['label', 'rand'])),
  description: 'Hula_S2_DynamicWorld_samples',
  fileFormat: 'CSV'
});

Export.image.toDrive({
  image: rfClass,
  description: 'Hula_RF_classification',
  region: aoiGeom,
  scale: 10,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: svmClass,
  description: 'Hula_SVM_classification',
  region: aoiGeom,
  scale: 10,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: label,
  description: 'Hula_DW_label_mode',
  region: aoiGeom,
  scale: 10,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: label,
  description: 'Hula_DW_label_mode_clean',
  region: aoiGeom,
  scale: 10,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: label.visualize({
    min: 0,
    max: 8,
    palette: [
      '419BDF', // 0 Water
      '397D49', // 1 Trees
      '88B053', // 2 Grass
      '7A87C6', // 3 Flooded vegetation
      'E49635', // 4 Crops
      'DFC35A', // 5 Shrub & scrub
      'C4281B', // 6 Built
      'A59B8F', // 7 Bare
      'B39FE1'  // 8 Snow & ice
    ]
  }),
  description: 'Hula_DW_label_visualized',
  region: aoiGeom,
  scale: 10,
  maxPixels: 1e13
});