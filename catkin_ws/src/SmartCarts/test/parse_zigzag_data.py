import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os

dir = os.path.dirname(__file__)
data_filename = os.path.join(dir, 'zigzag_1.csv')
df = pd.read_csv(data_filename, delimiter=',',header=None, names=['X','Y','R','T'], skiprows=52, skipfooter=129) # x,y,radius,timestamp

## CALIBRATION VARIABLES
## Apple Red Ball
actual_radius = 9.6 # cm - of red ball used
z_cal_power = 1.07 # usage: rel_dist^
z_cal_dist = 50 # cm
z_cal_pixels = 149.3 # pixel num
frame_center_x = 320 # pixel num
frame_center_y = 240 # pixel num

start_x = 0 # first frame, pixel num
start_y = 0 # first frame, pixel num
start_z = 0 # first frame, cm ## THIS UNIT IS DIFFERENT!! ##
est_x = [] # estimated x coord (relative)
est_y = [] # estimated y coord 
est_z = [] # estimated z coord (relative)

err_x = [] # error for each physical x point
err_z = [] # error for each physical z point

## PHYSICAL DATA
phys_x = [0, -50, 50, -50, 50]
phys_z = [0, 40, 90, 130, 170]
phys_t = [2000, 8000, 16000, 22000, 28000]

def cal_dist(radius_px):
	# Takes in radius pixel number and returns the distance
	return 10**((np.log10(z_cal_pixels)-np.log10(radius_px))/z_cal_power + np.log10(z_cal_dist))

def parse_data():
	# Outputs into csv file the x,y,radius from video. Artifical start defined by "skiprows"
	# Cycle through data
	for index, row in df.iterrows():
		# First step: Gets the distance between webcam and the plane that contains the centroid of the ball (estimated)
		## We dont care about the y value for this test
		z = cal_dist(row.R)
		if (index == 0):
			start_x = row.X
			start_y = row.Y 
			start_z = z

		# Second step: Gets the pixel/x or y cm @ z distance. If this is the first frame, also store the "original position"
		pixel_per_xy = row.R/actual_radius 

		# Third step: Get the x,y calculation
		est_x.append((row.X - frame_center_x)/pixel_per_xy)
		est_y.append((row.Y - frame_center_y)/pixel_per_xy) # aside: not much use including the relative position.
		est_z.append(cal_dist(row.R) - start_z)

	for i in range(len(phys_x)):
		t = phys_t[i]
		t_arr = df['T'].tolist() - t
		idx = t_arr.index(min(t_arr))
		err_x = df.loc['X',idx]
		print(err_x)
		x.append(err_x)

	# Plot Graph
	plt.figure(figsize=(10, 5))
	plt.title("Estimated Path based on Calibration Data")
	plt.xlabel("Time (ms)")
	plt.ylabel("Relative Distance from Start(cm)")
	size = [3 for n in range(len(est_z))]
	plt.scatter(df['T'], est_x, size)
	plt.scatter(df['T'], est_y, size)
	plt.scatter(df['T'], est_z, size)
	plt.scatter(phys_t, phys_x, marker='*')
	plt.scatter(phys_t, phys_z, marker='*')
	plt.legend(['est_x','est_y','est_z','actual_x','actual_z'])
	plt.show()

	# Save Data
	est_arr = np.asarray([est_x,est_y,est_z,df['T'].tolist()])

	filename = os.path.join(dir, 'parsed_zigzag_1.csv')
	np.savetxt(filename, np.transpose(est_arr), delimiter=",") 

	return

if __name__ == '__main__': 
	parse_data()
