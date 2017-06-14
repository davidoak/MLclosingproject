###################################################################
#
# Copyright (C) and Written by David Kao
# June of 2017
#
###################################################################


from azure.storage.blob import BlockBlobService
from azure.storage.blob import ContentSettings
import os, glob
import time


block_blob_service = BlockBlobService(account_name = 'YOUR_ACCOUNT_NAME',
				      account_key  = 'YOUR_ACCOUNT_KEY')

i = 0
ML_count = 0
blob_list = []

while True:

	i = i + 1
	print("\nIt is looping (awaiting) " + str(i) + " time(s) and ML_count = " + str(ML_count) + " so far.\n")

	# Listing all the exisitng blob 'allenclosing1' files if any
	print("\n------------LIST ALL FILES (If ANY)-----------\n")

	generator = block_blob_service.list_blobs('temp-images')
	for blob in generator:
	    print(blob.name)
	    if blob.name.endswith('.jpg'):
	    	blob_list = ["EXECUTE"]

	# Only executing the ML and the Deleting process after seeing the blob_list existance
	if blob_list != []:

		# Downloading all the images to local folder
		print("\n------------DOWNLOAD ALL FILES------------\n")

		for b in generator.items:
		    r = block_blob_service.get_blob_to_path('temp-images', b.name, "/home/mlclosingproject/py-faster-rcnn/data/demo/{}".format(b.name))

		# After downloading blob files, deleting it
		print("\n------------DELETING TEMP IMAGES------------\n")

		for b in generator.items:
			r = block_blob_service.delete_blob('temp-images', b.name)
			blob_list = []


	# Checking the local folder list
	filelist = glob.glob(os.path.join("/home/mlclosingproject/py-faster-rcnn/data/demo/", '*.jpg'))

	# If filelist contains any .jpg image, it executes further
	if filelist!=[]:

		# Calling Demo.py to do the ML part
		os.system("./demo.py --net=zf --cpu")
		time.sleep(2)
		ML_count = ML_count + 1

		# After demo.py being called, upload the tested images to the cloud
		print("\n------------UPLOAD TESTED IMAGES------------\n")

		for im_name in glob.glob("/home/mlclosingproject/py-faster-rcnn/data/david_output/chair/*.jpg"):
			print(im_name)
			block_blob_service.create_blob_from_path(
			    'davidclosing-chair',
			    os.path.basename(os.path.normpath(im_name)),
			    im_name,
			    content_settings=ContentSettings(content_type='image/jpg')
			)

		for im_name in glob.glob("/home/mlclosingproject/py-faster-rcnn/data/david_output/person/*.jpg"):
			print(im_name)
			block_blob_service.create_blob_from_path(
			    'davidclosing-person',
			    os.path.basename(os.path.normpath(im_name)),
			    im_name,
			    content_settings=ContentSettings(content_type='image/jpg')
			)		

		for im_name in glob.glob("/home/mlclosingproject/py-faster-rcnn/data/david_output/pottedplant/*.jpg"):
			print(im_name)
			block_blob_service.create_blob_from_path(
			    'davidclosing-pottedplant',
			    os.path.basename(os.path.normpath(im_name)),
			    im_name,
			    content_settings=ContentSettings(content_type='image/jpg')
			)

		for im_name in glob.glob("/home/mlclosingproject/py-faster-rcnn/data/david_output/tvmonitor/*.jpg"):
			print(im_name)
			block_blob_service.create_blob_from_path(
			    'davidclosing-tvmonitor',
			    os.path.basename(os.path.normpath(im_name)),
			    im_name,
			    content_settings=ContentSettings(content_type='image/jpg')
			)

		for im_name in glob.glob("/home/mlclosingproject/py-faster-rcnn/data/david_output/bottle/*.jpg"):
			print(im_name)
			block_blob_service.create_blob_from_path(
			    'davidclosing-bottle',
			    os.path.basename(os.path.normpath(im_name)),
			    im_name,
			    content_settings=ContentSettings(content_type='image/jpg')
			)

		# Deleting local images to prevent repeated testing
		print("\n------------DELETING LOCAL BEING TESTED FILES------------\n")

		for im_name in glob.glob("/home/mlclosingproject/py-faster-rcnn/data/demo/*.jpg"):
			print(im_name)
			os.remove(im_name)

		# Deleting local tested output to prevent same-file uploads
		print("\n------------DELETING LOCAL TESTED OUTPUT------------\n")

		for im_name in glob.glob("/home/mlclosingproject/py-faster-rcnn/data/david_output/*/*.jpg"):
			print(im_name)
			os.remove(im_name)


	time.sleep(3)
