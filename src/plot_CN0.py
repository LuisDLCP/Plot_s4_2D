#!/home/luis/anaconda3/bin/python3
#___________________________________________________________
#                  AMPLITUDE vs TIME 
#                        plots
#                        v1.0
#-----------------------------------------------------------
# This script creates Amplitude vs Time plots, for each 
# frequency and constellation. SBAS data is included in 
# GPS const by default. Elevation vs Time plots are also 
# included in the same graphs. Finally, these plots are 
# saved in an A4 pdf file.
# Author: Luis D.
# :)
from septentrio_tools import ProcessISMR, PlotsISMR
from matplotlib.backends.backend_pdf import PdfPages
import glob 
import os 

# Declare input and output paths 
root_path = "/home/luis/Desktop/Proyects_Files/LISN/GPSs/Tareas/Plot_s4_2D/"
input_files_path = root_path + "Input_data/Data_set2/"
input_files_path_op = root_path + "Input_data/Data_procesada/"
output_files_path = root_path + "Output_data/"
file_name = "ljic2810.20_.ismr" # Test file 

def process_dataframe(input_file):
    file1 = ProcessISMR()
    # Read and normalize the df
    file1.read_file(input_file)
    file1.normalize_df()
    # Rename the Elev column
    file1.rename_column(5, "Elev")
    # Rename the CN0 columns
    file1.rename_column(6, "CN0_sig1")
    file1.rename_column(31, "CN0_sig2")
    file1.rename_column(45, "CN0_sig3")
    # Extract certain columns 
    file1.extract_columns(cols=["Elev", "CN0_sig1", "CN0_sig2", "CN0_sig3"])
    # Convert to float
    file1.convert2float(cols=["Elev", "CN0_sig1", "CN0_sig2", "CN0_sig3"])
    # Filter df based on elevation col 
    for j in range(3):
        j += 1
        file1.filter_dataframe(col=f"CN0_sig{j}", on="Elev", threshold=35, new_col_name=[f"CN0_sig{j}_1", f"CN0_sig{j}_2"])

    df = file1.df
    print("df ready to plot!")

    return df 

def main():
    # Specify the const and freq to plot    
    const_list = ['G', 'E']   
    freq_list = ['CN0_sig1', 'CN0_sig2', 'CN0_sig3'] 

    list_input_files = glob.glob(input_files_path + "*.ismr")
    if len(list_input_files) > 0:
        for file_i in list_input_files:
            # Get the df
            df = process_dataframe(file_i)
            
            # Plot
            input_file_name = file_i[len(input_files_path):]
            g1 = PlotsISMR(dataframe=df, ismr_file_name=input_file_name)
            # -> Create an empty pdf file to save the plots
            figure_name = g1.get_output_figure_name() + "_CN0.pdf" # e.g. ljic_200806_CN0.pdf
            pdf = PdfPages(output_files_path + figure_name)
            # -> Generate the plots
            for c in const_list:
                for f in freq_list:         
                    g1.plotCN0(const=c, freq=f, pdf=pdf)
            pdf.close()

            # Move input files to a permanent directory
            os.rename(file_i, input_files_path_op + input_file_name)
    
    return 'Ok'

if __name__ == '__main__':
    main()
    print("FINISHED ----------")