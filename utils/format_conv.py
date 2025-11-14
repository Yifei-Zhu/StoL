from openbabel import openbabel as ob

def format_conv(input_file, output_file, input_format,output_format):
    # Create an OBConversion object
    conv = ob.OBConversion()
    # Set the input and output file formats
    conv.SetInAndOutFormats(input_format, output_format)
    # Open the input and output files
    conv.OpenInAndOutFiles(input_file, output_file)

    # Perform the format conversion (success of fail)
    sORf =  conv.Convert()
    # Close the output file
    conv.CloseOutFile()
    return sORf

if __name__ == '__main__':
    format_conv('test.xyz', 'test.sdf', 'sdf', 'xyz')
    