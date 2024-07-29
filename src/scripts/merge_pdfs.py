from PyPDF2 import PdfMerger
from pathlib import Path

def merge_pdfs_in_folder(folder_path, name):
    # Create a Path object for the folder
    folder = Path(folder_path)
    
    # Get a list of all PDF files in the folder and sort them alphabetically
    pdf_files = sorted(folder.glob('*.pdf'))
    
    # Initialize the PdfMerger object
    merger = PdfMerger()
    
    # Append each PDF file to the merger
    for pdf in pdf_files:
        merger.append(str(pdf))
        print(f'Appending: {pdf.name}')
    
    # Define the output path
    output_path = folder.parent / name
    
    # Write the merged PDF to the output path
    with open(output_path, 'wb') as output_pdf:
        merger.write(output_pdf)
    
    print(f'Merged PDF saved to: {output_path}')

if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent  # root folder of project
    merge_pdfs_in_folder(
        root / 'results' / 'svm rri' / 'test_imbalanced_cross' / 'plots_misclassifications' / 'COATDataset_COATDataset' / 'false negatives',
        'DiagnoStick_false_negatives.pdf'
    )
    merge_pdfs_in_folder(
        root / 'results' / 'svm rri' / 'test_imbalanced_cross' / 'plots_misclassifications' / 'COATDataset_COATDataset' / 'false positives',
        'DiagnoStick_false_positives.pdf'
    )
