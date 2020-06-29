library(leaflet)
library(sp)  # vector data
library(raster)  # raster data
library(rgdal)  # input/output, projections
library(rgeos)  # geometry ops
library(spdep)  # spatial dependence
library(dplyr)
library(tmap)
library(htmltools)


data <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\usgs_gages2.csv")

data <- data[ , -1]
label <- data.frame(data$site_no, data$dec_lat_va, data$dec_long_v)

## Dynamic Plot

m <- leaflet(data) %>%
  addCircleMarkers(~dec_long_v, ~dec_lat_va, 
                   radius=2, color= "navy", opacity = 0.5, 
                   popup = ~htmlEscape(label)) %>%
  addProviderTiles(providers$CartoDB.Positron) # basemap
m  # Print the map



# Rasters
library(raster)
library(sf)
va <- st_read("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\data\\VA_shape\\VA_shape\\GOVTUNIT_Virginia_State_Shape\\Shape\\GU_StateOrTerritory.shp")
va_proj <- st_transform(va, proj4string(dem))
crs(va_proj)


dem <- raster("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\data\\DEM\\dem_final.tif")

precip_5yr <- raster("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\data\\precip_rasters\\5yr_24ha.tif")
precip_5yr <- projectRaster(precip_5yr,crs=crs(va_proj))
precip_5yr <- crop(precip_5yr, va_proj)

precip_10yr <- raster("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\data\\precip_rasters\\10yr_24ha.tif")
precip_10yr <- projectRaster(precip_10yr,crs=crs(va_proj))
precip_10yr <- crop(precip_10yr, va_proj)

precip_25yr <- raster("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\data\\precip_rasters\\25yr_24ha.tif")
precip_25yr <- projectRaster(precip_25yr,crs=crs(va_proj))
precip_25yr <- crop(precip_25yr, va_proj)


slope <- raster("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\data\\slope\\slope_final.tif")

aspect <- raster("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\data\\aspect\\aspect\\aspect_final.tif")

ndmi <- raster("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\data\\NDMI\\ndmi_final.tif")

ndvi <- raster("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\data\\NDVI\\ndvi_final.tif")

land_cover <- raster("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\data\\land_cover\\land_cover_final.tif")

soil <- raster("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\data\\soil_type\\soil_final.tif")




# r <- raster("nc/oisst-sst.nc")
pal_aspect <- colorNumeric(c('#fff7fb','#ece2f0','#d0d1e6','#a6bddb','#67a9cf','#3690c0','#02818a','#016c59','#014636'), 
                           values(aspect),na.color = "transparent")
pal_dem <- colorNumeric(c('#fff7fb','#ece2f0','#d0d1e6','#a6bddb','#67a9cf','#3690c0','#02818a','#016c59','#014636'), 
                           values(dem),na.color = "transparent")
pal_landcover <- colorNumeric(c('#fff7fb','#ece2f0','#d0d1e6','#a6bddb','#67a9cf','#3690c0','#02818a','#016c59','#014636'), 
                           values(land_cover), na.color = "transparent")
pal_ndmi <- colorNumeric(c('#fff7fb','#ece2f0','#d0d1e6','#a6bddb','#67a9cf','#3690c0','#02818a','#016c59','#014636'), 
                           values(ndmi),na.color = "transparent")
pal_ndvi <- colorNumeric(c('#fff7fb','#ece2f0','#d0d1e6','#a6bddb','#67a9cf','#3690c0','#02818a','#016c59','#014636'), 
                           values(ndvi),na.color = "transparent")
pal_precip5yr <- colorNumeric(c('#fff7fb','#ece2f0','#d0d1e6','#a6bddb','#67a9cf','#3690c0','#02818a','#016c59','#014636'), 
                           values(precip_5yr),na.color = "transparent")
pal_precip10yr <- colorNumeric(c('#fff7fb','#ece2f0','#d0d1e6','#a6bddb','#67a9cf','#3690c0','#02818a','#016c59','#014636'), 
                           values(precip_10yr),na.color = "transparent")
pal_precip25yr <- colorNumeric(c('#fff7fb','#ece2f0','#d0d1e6','#a6bddb','#67a9cf','#3690c0','#02818a','#016c59','#014636'), 
                           values(precip_25yr),na.color = "transparent")
pal_slope <- colorNumeric(c('#fff7fb','#ece2f0','#d0d1e6','#a6bddb','#67a9cf','#3690c0','#02818a','#016c59','#014636'), 
                           values(slope),na.color = "transparent")
pal_soil <- colorNumeric(c('#fff7fb','#ece2f0','#d0d1e6','#a6bddb','#67a9cf','#3690c0','#02818a','#016c59','#014636'), 
                           values(soil), na.color = "transparent")





leaflet() %>% addTiles() %>%
  addRasterImage(aspect, colors = pal_aspect, opacity = 0.8) %>%
  addLegend(pal = pal_aspect, values = values(aspect),
            title = "Aspect")
leaflet() %>% addTiles() %>%
  addRasterImage(dem, colors = pal_dem, opacity = 0.8) %>%
  addLegend(pal = pal_dem, values = values(dem),
            title = "DEM")
leaflet() %>% addTiles() %>%
  addRasterImage(land_cover, colors = pal_landcover, opacity = 0.8) %>%
  addLegend(pal = pal_landcover, values = values(land_cover),
            title = "Land Cover")
leaflet() %>% addTiles() %>%
  addRasterImage(ndmi, colors = pal_ndmi, opacity = 0.8) %>%
  addLegend(pal = pal_ndmi, values = values(ndmi),
            title = "NDMI")
leaflet() %>% addTiles() %>%
  addRasterImage(ndvi, colors = pal_ndvi, opacity = 0.8) %>%
  addLegend(pal = pal_ndvi, values = values(ndvi),
            title = "NDVI")
leaflet() %>% addTiles() %>%
  addRasterImage(precip_5yr, colors = pal_precip5yr, opacity = 0.8) %>%
  addLegend(pal = pal_precip5yr, values = values(precip_5yr),
            title = "Precipiation: 5 year")
leaflet() %>% addTiles() %>%
  addRasterImage(precip_10yr, colors = pal_precip10yr, opacity = 0.8) %>%
  addLegend(pal = pal_precip10yr, values = values(precip_10yr),
            title = "Precipitation: 10 year")
leaflet() %>% addTiles() %>%
  addRasterImage(precip_25yr, colors = pal_precip25yr, opacity = 0.8) %>%
  addLegend(pal = pal_precip25yr, values = values(precip_25yr),
            title = "Precipitation: 25 year")
leaflet() %>% addTiles() %>%
  addRasterImage(slope, colors = pal_slope, opacity = 0.8) %>%
  addLegend(pal = pal_slope, values = values(slope),
            title = "Slope")
leaflet() %>% addTiles() %>%
  addRasterImage(soil, colors = pal_soil, opacity = 0.8) %>%
  addLegend(pal = pal_soil, values = values(soil),
            title = "Soil")





# Interactive Layer Display
# outline <- quakes[chull(quakes$long, quakes$lat),]



map <- leaflet(data) %>%
  # Base groups
  addCircleMarkers(~dec_long_v, ~ dec_lat_va,
                   radius=2, color="navy", opacity = 0.5, group = "Gages") %>% # Gages
  addProviderTiles(providers$CartoDB.Positron, group= "Basemap") %>% # Basemap
  
  # Overlay groups
  # addCircles(~long, ~lat, ~10^mag/5, stroke = F, group = "Quakes") %>%
  # addPolygons(data = outline, lng = ~long, lat = ~lat,
  #           fill = F, weight = 2, color = "#FFFFCC", group = "Outline") %>%

  # addRasterImage(aspect, colors = pal_aspect, opacity = 0.8, group= "Aspect") %>%
  addRasterImage(dem, colors = pal_dem, opacity = 0.8, group = "DEM") %>%
  # addRasterImage(land_cover, colors = pal_landcover, opacity = 0.8, group="Land Cover") %>%
  # addRasterImage(ndmi, colors = pal_ndmi, opacity = 0.8, group="NDMI") %>%
  addRasterImage(ndvi, colors = pal_ndvi, opacity = 0.8, group = "NDVI") %>%
  #addRasterImage(precip_5yr, colors = pal_precip5yr, opacity = 0.8, group="Precip 5 yr") %>%
  #addRasterImage(precip_10yr, colors = pal_precip10yr, opacity = 0.8, group = "Precip 10 yr") %>%
  #addRasterImage(precip_25yr, colors = pal_precip25yr, opacity = 0.8, group = "Precip 25 yr") %>%
  #addRasterImage(slope, colors = pal_slope, opacity = 0.8, group= "Slope") %>%
  # addRasterImage(soil, colors = pal_soil, opacity = 0.8, group = "Soil") %>%

    # Layers control
  addLayersControl(
    baseGroups = c("Gages", "Basemap"),
    overlayGroups = c("Aspect", "DEM", "Land Cover", 
                      "NDMI", "NDVI", "Precip 5 yr", 
                      "Precip 10 yr", "Precip 25 yr", 
                      "Slope", "Soil"),
    options = layersControlOptions(collapsed = TRUE)
  )
map



