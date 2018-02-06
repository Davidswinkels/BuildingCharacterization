## Load modules
import os, sys, shutil
import pandas
import numpy as np

## Load workspace
os.chdir("D:\Workspace\Scripts")

## Set parameters
input_file = "./input/BuildingPointsValid.csv"
training_perc = 60.0
validation_perc = 20.0
testing_perc = 20.0
output_file = "D:\\Workspace\\Data\\streetview-master\\data_valid_resid_any"

## Load data
building_points = pandas.read_csv(input_file)

## Check data
print list(building_points)


# Clear data_valid_resid map
## Save images per fov in training, validation and testing repositories 
for fovlevel in ["30", "60", "90"]:
    for resid_info in ["\\residential", "\\non_residential"]:
        for dest_info in ["\\training", "\\validation", "\\testing"]:
            dest = output_file + "\\F" + fovlevel + resid_info + dest_info 
            if os.path.exists(dest):
                shutil.rmtree(dest, ignore_errors=True)    
            if not os.path.exists(dest):
                os.makedirs(dest)     

## Select valid points only (valid = 'Yes')
building_points_valid = building_points.loc[building_points['valid'] == 'Yes']

## Find panorama with no images
buildings_unvalid = []
for idx, building in building_points_valid.iterrows():
    filename = "N" + building['BU_CODE'] + "_B" + str(building['BuildingID']) + "_P" + building['pano_id'] + "_F30_A00.jpg"
    source_info = "\\" + building['BU_CODE'][-4:] + "\\" + filename
    source = "D:\\Workspace\\Data\\streetview-master\\data" + source_info
    if os.path.isfile(source) == False:
        buildings_unvalid += [building['BuildingID']]
    print "Source: " + source
    

## Remove panorama point with no images
bu_points_rem = building_points_valid[~building_points_valid['BuildingID'].isin(buildings_unvalid)]

bu_points_rem['Class'] = bu_points_rem['Residentia'].map(str) + bu_points_rem['Meeting'].map(str) + bu_points_rem['Healthcare'].map(str) + bu_points_rem['Industry'].map(str) + bu_points_rem['Office'].map(str) + bu_points_rem['Accommodat'].map(str) + bu_points_rem['Education'].map(str) + bu_points_rem['Sport'].map(str) + bu_points_rem['Shop'].map(str) + bu_points_rem['Other'].map(str)

## Select only useful columns ID, BU_CODE, BuildingID, Residentia
images_valid = bu_points_rem.loc[:, ['ID', 'BuildingID', 'BU_CODE', 'pano_id', 'Class', 'Residentia']]
##print images_valid
images_valid['Resident'] = np.where(images_valid['Class']=='1000000000', 1, 0)

## Check distribution of residential images per BU_CODE
# Check distribution of buildings per neighbourhood
building_distr = images_valid.groupby('BU_CODE')['BU_CODE'].count()
##print building_distr

# Check distribution of residential buildings per neighbourhood
residential_building_distr = images_valid.groupby('BU_CODE')['Residentia'].sum()
##print residential_building_distr

# Check distribution of non-residential buildings per neighbourhood
non_residential_building_distr = building_distr - residential_building_distr
##print non_residential_building_distr

## Create training, validation and testing datasets with similar distribution
## Instantiate empty variables
training_neigh = []
validation_neigh = []
testing_neigh = []
training_residential = 0.0
training_non_residential = 0.0
validation_residential = 0.0
validation_non_residential = 0.0
testing_residential = 0.0
testing_non_residential = 0.0

## Create list of neighbourhood codes
neighbourhood_codes = images_valid['BU_CODE'].unique()

## Loop over all neighbourhoods to assigin every neighbourhood to a training, validation or testing dataset
for idx, neigh in enumerate(neighbourhood_codes):
    print "Iteration number: " + str(idx + 1)
    # Instantiate training, validation and testing datasets with 5 neighbourhoods each
    if len(training_neigh) < 5:
        training_neigh.append(neigh)
        training_residential += float(residential_building_distr[idx])
        training_non_residential += float(non_residential_building_distr[idx])
        training_total = training_non_residential + training_residential
        training_resid_perc = (training_residential / training_total) * 100.0
    elif len(validation_neigh) < 5:
        validation_neigh.append(neigh)
        validation_residential += float(residential_building_distr[idx])
        validation_non_residential += float(non_residential_building_distr[idx])
        validation_total = validation_non_residential + validation_residential
        validation_resid_perc = (validation_residential / validation_total) * 100.0
    elif len(testing_neigh) < 5:
        testing_neigh.append(neigh)
        testing_residential += float(residential_building_distr[idx])
        testing_non_residential += float(non_residential_building_distr[idx])
        testing_total = testing_non_residential + testing_residential
        testing_resid_perc = (testing_residential / testing_total) * 100.0
    if idx >= 15:
        building_total = training_total + validation_total + testing_total     
        # Fill training, validation and testing dataset to have a respective 70%, 15% and 15% distribution of data
        if ((float(training_total) / float(building_total)) * 100.0) < training_perc:
            training_neigh.append(neigh)
            training_residential += float(residential_building_distr[idx])
            training_non_residential += float(non_residential_building_distr[idx])
            training_total = training_non_residential + training_residential
            training_resid_perc = (training_residential / training_total) * 100.0
##            print "To training:" + str((float(training_total) / float(building_total)) * 100.0)
##            print training_resid_perc
        elif ((float(validation_total) / float(building_total)) * 100.0) < validation_perc:          
            validation_neigh.append(neigh)
            validation_residential += float(residential_building_distr[idx])
            validation_non_residential += float(non_residential_building_distr[idx])
            validation_total = validation_non_residential + validation_residential
            validation_resid_perc = (validation_residential / validation_total) * 100.0
##            print "To validation:" + str((float(validation_total) / float(building_total)) * 100.0)
##            print validation_resid_perc
        elif ((float(testing_total) / float(building_total)) * 100.0) < testing_perc:
            testing_neigh.append(neigh)
            testing_residential += float(residential_building_distr[idx])
            testing_non_residential += float(non_residential_building_distr[idx])
            testing_total = testing_non_residential + testing_residential
            testing_resid_perc = (testing_residential / testing_total) * 100.0
##            print "To testing:" + str((float(testing_total) / float(building_total)) * 100.0)
##            print testing_resid_perc
        print "Total number of buildings added " + str(int(building_total) + residential_building_distr[idx] + non_residential_building_distr[idx]) + " out of " + str(sum(building_distr)) + " buildings"

## Add information of training, validation and testing to pandas dataframe of images
images_valid['type'] = "Undefined"
for idx, image in images_valid.iterrows():
    if image['BU_CODE'] in training_neigh:
        images_valid.at[int(image['ID']),'type'] = "Training"
    if image['BU_CODE'] in validation_neigh:
        images_valid.at[int(image['ID']),'type'] = "Validation"
    if image['BU_CODE'] in testing_neigh:
        images_valid.at[int(image['ID']),'type'] = "Testing"

        
## Save images per fov in training, validation and testing repositories 
for fovlevel in ["30", "60", "90"]:
    for idx, image in images_valid.iterrows():
        resid_info = "\\residential"
        if image['Residentia'] == 0:
            resid_info = "\\non_residential"
        if image['type'] == "Training":
            filename = "\\N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" + image['pano_id'] + "_F" + fovlevel + "_A00.jpg"
            source_info = "\\" + image['BU_CODE'][-4:] + "\\" + filename
            dest_info = resid_info + "\\training" + filename
        if image['type'] == "Validation":
            filename = "\\N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" + image['pano_id'] + "_F" + fovlevel + "_A00.jpg"
            source_info = "\\" + image['BU_CODE'][-4:] + "\\" + filename
            dest_info = resid_info + "\\validation" + filename
        if image['type'] == "Testing":
            filename = "\\N" + image['BU_CODE'] + "_B" + str(image['BuildingID']) + "_P" + image['pano_id'] + "_F" + fovlevel + "_A00.jpg"
            source_info = "\\" + image['BU_CODE'][-4:] + "\\" + filename
            dest_info = resid_info + "\\testing" + filename   
        source = "D:\\Workspace\\Data\\streetview-master\\data" + source_info
        destination = output_file + "\\F" + fovlevel + dest_info
        print "Source:" + source
        print "Destination:" + destination
        shutil.copyfile(source, destination)

    
