from definitions import *

def order_subsampled_df(larger_df_file, smaller_df_file):
	# Load the larger CSV into a dictionary
	larger_csv_path = f'{larger_df_file}.csv'
	smaller_csv_path = f'{smaller_df_file}.csv'

	# Load CSV files
	larger_df = pd.read_csv(larger_csv_path)
	smaller_df = pd.read_csv(smaller_csv_path)

	# Create keys for matching
	larger_df['key'] = larger_df[['Points:0', 'Points:1', 'Points:2']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
	smaller_df['key'] = smaller_df[['X', 'Y', 'Z']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

	# Create a dictionary for the order based on the larger_df
	order_dict = {k: i for i, k in enumerate(larger_df['key'])}

	# Add an 'order' column to smaller_df based on the larger_df's order
	smaller_df['order'] = smaller_df['key'].map(order_dict)

	# Drop rows without a matching key in larger_df (if necessary)
	smaller_df = smaller_df.dropna(subset=['order'])

	# Sort the smaller_df based on the new 'order' column
	smaller_df_sorted = smaller_df.sort_values(by='order')

	# Remove temporary columns before saving
	smaller_df_final = smaller_df_sorted.drop(columns=['key', 'order'])

	ordered_smaller_df_file = f'{smaller_df_file}_ordered'

	# Save the reordered smaller CSV
	smaller_df_final.to_csv(f'{ordered_smaller_df_file}.csv', index=False)

	return ordered_smaller_df_file

def subsample_all_files(ordered_df_filename, save_filename):

	smaller_df_final = pd.read_csv(f'{ordered_df_filename}.csv')

	# Assuming smaller_df_final is your reordered smaller_df with the correct columns
	smaller_df_dict = {
	    (row['X'], row['Y'], row['Z']): row
	    for index, row in smaller_df_final.iterrows()
	}


	def subsample_larger_df(filename, smaller_df_dict):
		larger_df = pd.read_csv(filename)
		wind_angle = int(filename.split('_')[-1].split('.')[0])

		# Convert Points:0, Points:1, Points:2 to a uniform key format
		larger_df['key'] = larger_df[['Points:0', 'Points:1', 'Points:2']].apply(lambda row: (row['Points:0'], row['Points:1'], row['Points:2']), axis=1)

		# Filter rows based on the key existence in the smaller_df_dict
		subsampled_df = larger_df[larger_df['key'].map(lambda x: x in smaller_df_dict)]

		# Optionally, drop the 'key' column if no longer needed
		subsampled_df = subsampled_df.drop(columns=['key'])

		new_filename = f'{save_filename}_{wind_angle}'

		subsampled_df.to_csv(f'{new_filename}.csv')

		return new_filename

	# Subsample each DataFrame in the list
	subsampled_dfs = [subsample_larger_df(filename, smaller_df_dict) for filename in filenames]

	return subsampled_dfs

ordered_df_filename = order_subsampled_df('CFD_cell_data_simulation_0', 'Sampled_70_70_10_CFD_cell_data_simulation_0')
filenames = get_filenames_from_folder('.', '.csv', 'CFD_cell_data_simulation_')
new_filenames = subsample_all_files(ordered_df_filename, 'Sampled_70_70_10_CFD_cell_data_simulation_new')

print (new_filenames)