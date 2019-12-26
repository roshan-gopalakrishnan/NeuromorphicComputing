Training code for HFNet, written in Tensorpack. 

Steps to follow:
1. Install Tensorpack.
2. Replace the conv2d.py file from the Tensorpack code package.
3. Download the IMAGENET dataset.
4. Run the HFNet.py code.
python HFNet.py --data "path_to_IMAGENET_folder" --gpu 0,1,2,3
