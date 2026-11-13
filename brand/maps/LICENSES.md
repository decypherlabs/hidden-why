# Map data

- `world.json`, `india.json`: Natural Earth country boundaries (public domain), India's own boundary.
- `terrain_10800.jpg`: Natural Earth "Natural Earth I with Shaded Relief and Water" (NE1_50M_SR_W), 10800x5400,
  re-saved from the original GeoTIFF as JPEG q88 (8.4 MB). Source: https://www.naturalearthdata.com/downloads/50m-raster-data/50m-natural-earth-1/
  (https://naturalearth.s3.amazonaws.com/50m_raster/NE1_50M_SR_W.zip). Licence: public domain, no attribution required
  (https://www.naturalearthdata.com/about/terms-of-use/). Equirectangular, WGS84, 1/30 degree per pixel.
- It is graded at load time by `kit/maps.py` and `MapShot` in `pipeline/build.py` (MAPS_TERRAIN=0 turns the terrain off).

- places.json: Natural Earth 1:50m populated places and marine polygons (public domain, naturalearthdata.com), city and sea label points.
