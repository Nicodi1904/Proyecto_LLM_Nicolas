import zipfile
import xml.etree.ElementTree as ET
import sys
import os

def get_docx_text(path, output_path):
    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    
    try:
        document = zipfile.ZipFile(path)
        xml_content = document.read('word/document.xml')
        document.close()
        
        tree = ET.XML(xml_content)
        
        paragraphs = []
        
        # Word XML namespace required to find tags
        WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        PARA = WORD_NAMESPACE + 'p'
        TEXT = WORD_NAMESPACE + 't'
        
        for p in tree.iter(PARA):
            texts = [node.text for node in p.iter(TEXT) if node.text]
            if texts:
                paragraphs.append(''.join(texts))
        
        full_text = '\n\n'.join(paragraphs)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
            
        return f"Successfully wrote to {output_path}"
    except Exception as e:
        return f"Error reading {path}: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python read_docx.py <input_file> <output_file>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    print(get_docx_text(input_path, output_path))
