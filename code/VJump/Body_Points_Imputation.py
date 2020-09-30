import numpy as np
import pandas as pd

def find_break_in_series(points_series, start_i):
	
	break_start_i = start_i - 1
	break_end_i = start_i
	break_found = False
	continue_crawling = True
	
	while (continue_crawling == True) and (break_end_i < points_series.index[-1]):
	
		if points_series[break_end_i] != None:
			break_found = False
			break_start_i = break_end_i
			break_end_i += 1
			
		else:
			break_found = True
			# break_end_i += 1
			
			while (break_found == True) and (break_end_i <= points_series.index[-1]):
				
				if points_series[break_end_i] == None:
					break_end_i += 1
				else:
					break_found = False
					
			continue_crawling = False
		
	return break_start_i, break_end_i

def impute_break_in_series(points_series, start_i, end_i):
	
	diff = end_i - start_i
	start_x, start_y = points_series[start_i]
	end_x, end_y = points_series[end_i]
	
	diff_x = (end_x - start_x) / diff
	diff_y = (end_y - start_y) / diff
	
	multiplier = 1
	
	for i in range(start_i+1, end_i):
		x = int(round(start_x + (diff_x * multiplier)))
		y = int(round(start_y + (diff_y * multiplier)))
		points_series[i] = (x, y)
		multiplier += 1

def imputing_missing_points_in_series(points_series):
	
	series_start_i = points_series.index[0]
	series_end_i = points_series.index[-1]
	
	current_i = series_start_i
	
	while current_i < series_end_i:
		
		if points_series[current_i] == None:
			current_i += 1
			
		else:
			break_start_i, break_end_i = find_break_in_series(points_series, current_i)
			
			if break_end_i - break_start_i == 1:
				break
			elif break_end_i > series_end_i:
				break
			else: 
				impute_break_in_series(points_series, break_start_i, break_end_i)
				current_i = break_end_i
	
	return points_series