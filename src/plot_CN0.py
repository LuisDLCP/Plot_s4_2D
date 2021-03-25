from septentrio_tools import ProcessISMR, PlotsISMR
from matplotlib.backends.backend_pdf import PdfPages

# Declare variables
root_path = "/home/luis/Desktop/Proyects_Files/LISN/GPSs/Tareas/Graficas_desvanecimientos/"
input_files_path = root_path + "Input_data/Data_set/"
input_files_path_op = root_path + "Input_data/Data_procesada/"
output_files_path = root_path + "Output_data/"
file_name = "ljic2810.20_.ismr"

# Processing
file1 = ProcessISMR()
# Read and normalize the df
file1.read_file(input_files_path + file_name)
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

# Test
df = file1.df
print(df.head())
print(list(df))
#print(df.info())
#print(df.info())
print("df ready to plot!")
#file1.plot_fast("Elev")
#const = file1.check_constelations()
#print(const)

# Plotting
const_list = ['G', 'E'] #  Constelations list  
freq_list = ['CN0_sig1', 'CN0_sig2', 'CN0_sig3'] # Frecuencies list 

g1 = PlotsISMR(dataframe=df, ismr_file_name=file_name)
#print(g1.df.head())

figure_name2 = g1._get_output_figure_name() + "_CN0.pdf" # e.g. ljic_200806_CN0.pdf
pdf = PdfPages(output_files_path + figure_name2)

for c in const_list:
    for f in freq_list:         
        g1.plotCN0(const=c, freq=f, pdf=pdf, sbas=True)

pdf.close()
print("Finish")

# # -> Specify the consts and freqs to plot 
# const_list = ['G', 'E'] #  Constelations list  
# freq_list = ['S4_sig1', 'S4_sig2', 'S4_sig3'] # Frecuencies list 

# list_input_files = glob.glob(input_files_path + "*.s4")
# if len(list_input_files) > 0:
#     for file_i in list_input_files:
#         g = ScintillationPlot()
#         g.read_s4_file(file_i)
#         g.process_dataframe()
#         g.filter_dataframe() # Dataframe ready to plot 

#         # Plot 
#         # -> Create an empty pdf file to save the plots
#         figure_name2 = g.figure_name() + "_CN0.pdf" # e.g. ljic_200806_s4.pdf
#         pdf = PdfPages(output_files_path + figure_name2)

#         # -> Generate the plots 
#         for c in const_list:
#             for f in freq_list:         
#                 fig = g.plot1_s4(const=c, freq=f, sbas=True, pdf=pdf) 
#         pdf.close()
        
#         # Move input files to a permanent directory
#         file_name = file_i[len(input_files_path):]
#         os.rename(file_i, input_files_path_op+file_name)