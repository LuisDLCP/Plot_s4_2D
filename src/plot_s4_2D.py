from matplotlib.backends.backend_pdf import PdfPages
import septentrio_tools as st 
import glob
import os 

# Declare input and output paths 
root_path = "/home/luis/Desktop/Proyects_Files/LISN/GPSs/Tareas/Plot_s4_2D/"
input_files_path = root_path + "Input_data/Data_set/"
input_files_path_op = root_path + "Input_data/Data_procesada/"
output_files_path = root_path + "Output_data/"
#file_name = "ljic2810.20_.ismr" # Test file 

def merge_df():
    """
    Merge s4 dataframes from many days 
    """
    list_files = glob.glob(input_files_path + "*.ismr")
    list_files = sorted(list_files, key=lambda x: (x[-8:-6], x[-17:-9])) # sort by year first, then by doy  
    j = 0
    if len(list_files) > 0:
        for archivo in list_files:
            file1 = st.ProcessISMR()
            file1.read_file(archivo)
            df1 = file1.get_s4()

            if j == 0:
                df2 = df1
            else:
                # Merge dfs
                df2 = df2.append(other=df1)
            j += 1  

            # Move input files to a permanent directory
            #input_file_name = archivo[len(input_files_path):]
            #os.rename(archivo, input_files_path_op + input_file_name)

    return df2 

def main():
    # Specify the const and freq to plot    
    const_list = ['G', 'E']   
    freq_list = ['S4_sig1', 'S4_sig2', 'S4_sig3'] 

    # Get merged df 
    df = merge_df()

    # Plot
    pdf = PdfPages(output_files_path + "s4_months.pdf")
    for const in const_list:
        for freq in freq_list:
            m = st.PlotsISMR(dataframe=df)
            m.plotS4_2D(pdf=pdf, const=const, freq=freq)
    pdf.close()

    return 'Ok'

if __name__ == '__main__':
    print("STARTING ...")
    main()
    print("FINISHED ----------")