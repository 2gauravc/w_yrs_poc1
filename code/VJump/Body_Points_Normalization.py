import numpy as np
import pandas as pd

def find_base_normal_length(points_df, start_point, end_point):
	
	base_normal_length = 0
	
	for i in points_df.index:
		
		if ((points_df.loc[i, start_point] != None) and (points_df.loc[i, end_point] != None)):
			
			start_point_x, start_point_y = points_df.loc[i, start_point]
			end_point_x, end_point_y = points_df.loc[i, end_point]
			
			base_normal_length = int(round(np.sqrt((start_point_x - end_point_x)**2 + (start_point_y - end_point_y)**2)))
			
			break
			
	return base_normal_length

def normalizing_body_points_in_df(points_df, base_normal_start, base_normal_end, base_normal_mult, origin_point):
	
	base_normal_length = find_base_normal_length(points_df, base_normal_start, base_normal_end)
	
	points_norm_df = points_df.copy()
	points_norm_df["normalized?"] = "no"
	points_norm_df["origin_point"] = origin_point

	for i in points_norm_df.index:
		
		if points_norm_df.loc[i, origin_point] != None:
			points_norm_df.loc[i, "normalized?"] = "yes"
			origin_x, origin_y = points_norm_df.loc[i, origin_point]
	
			for p in [point for point in points_norm_df.columns if point not in ["normalized?", "origin_point"]]:
				if p == origin_point:
					norm_x = 0.0
					norm_y = 0.0
					points_norm_df.at[i, p] = (norm_x, norm_y)
					
				elif points_norm_df.loc[i, p] == None:
					pass
				
				else:
					p_x, p_y = points_norm_df.loc[i, p]
					norm_x = (p_x - origin_x) / (base_normal_length * base_normal_mult)
					norm_y = (p_y - origin_y) / (base_normal_length * base_normal_mult)
					# norm_y = -1 * norm_y
					points_norm_df.at[i, p] = (norm_x, norm_y)
			
	return points_norm_df