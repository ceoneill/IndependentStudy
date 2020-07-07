aspect <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\aspect\\aspect_values.csv")
dem <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\DEM\\dem_values.csv")
land_cover <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\land_cover\\land_cover_values.csv")
ndmi <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\NDMI\\ndmi_values.csv")
ndvi <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\NDVI\\ndvi_values.csv")
precip_5yr <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\precip_rasters\\precip_5yr_values.csv")
precip_10yr <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\precip_rasters\\precip_10yr_values.csv")
precip_25yr <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\precip_rasters\\precip_25yr_values.csv")
slope <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\slope\\slope_values.csv")
soil <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\soil_type\\soil_type_values.csv")


names(aspect)[names(aspect) == 'RASTERVALU'] <- 'aspect_value'
names(dem)[names(dem) == 'RASTERVALU'] <- 'dem_value'
names(land_cover)[names(land_cover) == 'RASTERVALU'] <- 'land_cover_value'
names(ndmi)[names(ndmi) == 'RASTERVALU'] <- 'ndmi_value'
names(ndvi)[names(ndvi) == 'RASTERVALU'] <- 'ndvi_value'
names(precip_5yr)[names(precip_5yr) == 'RASTERVALU'] <- 'precip_5yr_value'
names(precip_10yr)[names(precip_10yr) == 'RASTERVALU'] <- 'precip_10yr_value'
names(precip_25yr)[names(precip_25yr) == 'RASTERVALU'] <- 'precip_25yr_value'
names(slope)[names(slope) == 'RASTERVALU'] <- 'slope_value'
names(soil)[names(soil) == 'RASTERVALU'] <- 'soil_value'



usgs_gages <- cbind(aspect, dem$dem_value,
                    land_cover$land_cover_value, ndmi$ndmi_value,
                    ndvi$ndvi_value, precip_5yr$precip_5yr_value,
                    precip_10yr$precip_10yr_value, precip_25yr$precip_25yr_value,
                    slope$slope_value, soil$soil_value)

names(usgs_gages)[names(usgs_gages) == 'dem$dem_value'] <- 'dem_value'
names(usgs_gages)[names(usgs_gages) == 'land_cover$land_cover_value'] <- 'land_cover_value'
names(usgs_gages)[names(usgs_gages) == 'ndmi$ndmi_value'] <- 'ndmi_value'
names(usgs_gages)[names(usgs_gages) == 'ndvi$ndvi_value'] <- 'ndvi_value'
names(usgs_gages)[names(usgs_gages) == 'precip_5yr$precip_5yr_value'] <- 'precip_5yr_value'
names(usgs_gages)[names(usgs_gages) == 'precip_10yr$precip_10yr_value'] <- 'precip_10yr_value'
names(usgs_gages)[names(usgs_gages) == 'precip_25yr$precip_25yr_value'] <- 'precip_25yr_value'
names(usgs_gages)[names(usgs_gages) == 'slope$slope_value'] <- 'slope_value'
names(usgs_gages)[names(usgs_gages) == 'soil$soil_value'] <- 'soil_value'


#write.csv(usgs_gages,"C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\usgs_gages.csv", row.names = TRUE)


# ngvd29 <- which(usgs_gages$alt_datum_ == 'NGVD29')
# ngvd29
# 
# NGVD29 = NAVD88 - 3.6ft

usgs_gages2 <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\usgs_gages2.csv")


########

# test <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\gage_height\\sitedata_371631079542600.csv")
# 
# min(test$X_00065_00000)
# max(test$X_00065_00000)
# median(test$X_00065_00000)

gage_height <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\site_stats.csv")

names(gage_height)[names(gage_height) == 'sitenumber'] <- 'site_no'
gage_height <- gage_height[-207, ]



usgs_gages2 <- cbind(usgs_gages2, gage_height$min, gage_height$max, gage_height$median)
usgs_gages2[ ,"flood"] <- NA
View(usgs_gages2)

# usgs_gages2 <- usgs_gages2[ , -c(1, 20:26)]
write.csv(usgs_gages2,"C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\usgs_gages2.csv", row.names = TRUE)

which(usgs_gages2 == '7,17')



###############
test <- read.csv("\\\\surly.mcs.local\\Flood\\Temp\\coneill\\CRRA_data.csv")
View(test)
