from matplotlib.backends.backend_pdf import PdfPages
import septentrio_tools as st 
import glob

# Declare input and output paths 
root_path = "/home/luis/Desktop/Proyects_Files/LISN/GPSs/Tareas/Plot_s4_2D/"
input_files_path = root_path + "Input_data/Data_set/"
input_files_path_op = root_path + "Input_data/Data_procesada/"
output_files_path = root_path + "Output_data/"
file_name = "ljic2810.20_.ismr" # Test file 

# Get s4 dataframe (test)
print("Starting ...")

# Merge s4 dataframes from many days 
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

print(df2.head())

pdf = PdfPages("s4_2D_test3_filtered.pdf")
m = st.PlotsISMR(dataframe=df2, ismr_file_name=file_name)
m.plotS4_2D(pdf=pdf, const='G', freq='S4_sig2')
pdf.close()
