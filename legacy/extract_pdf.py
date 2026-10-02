import pdfplumber
import os

pdf_filename = "112學年 資工系課規.pdf" # Make sure this matches your file name
output_filename = "output_new.md"

def table_to_markdown(table):
    md_lines = []
    
    cleaned_table = []
    for row in table:
        cleaned_row = []
        for cell in (row or []):
            if cell is None:
                cleaned_row.append("")
            else:
                cleaned_row.append(str(cell).replace("\n", "").replace("\r", " ").strip())
        cleaned_table.append(cleaned_row)
        
    if not cleaned_table:
        return ""
        
    # Generate markdown table
    md_lines.append("| " + " | ".join(cleaned_table[0]) + " |")
    md_lines.append("|" + "|".join(["---"] * len(cleaned_table[0])) + "|")
    
    for row in cleaned_table[1:]:
        if len(row) < len(cleaned_table[0]):
            row = row + [""] * (len(cleaned_table[0]) - len(row))
        md_lines.append("| " + " | ".join(row[:len(cleaned_table[0])]) + " |")
        
    return "\n".join(md_lines) + "\n"

def not_within_bboxes(obj, bboxes):
    """Check if an object (e.g., char) is inside any of the given bounding boxes."""
    if obj["object_type"] != "char":
        return True
    
    obj_x_mid = (obj["x0"] + obj["x1"]) / 2
    obj_y_mid = (obj["top"] + obj["bottom"]) / 2
    
    for bbox in bboxes:
        x0, top, x1, bottom = bbox
        if x0 <= obj_x_mid <= x1 and top <= obj_y_mid <= bottom:
            return False
            
    return True

print(f"Extracting '{pdf_filename}'...")

try:
    with pdfplumber.open(pdf_filename) as pdf:
        with open(output_filename, "w", encoding="utf-8") as f:
            for i, page in enumerate(pdf.pages):
                f.write(f"## Page {i+1}\n\n")
                
                # --- 新增：物理裁切邏輯 ---
                # 建立邊界框：左、上、右、下各往內縮 40 points，切掉最外圍的黑框
                crop_bbox = (40, 40, page.width - 40, page.height - 40)
                # 產生新的裁切頁面物件
                cropped_page = page.crop(crop_bbox)
                
                # --- 注意：接下來的 find_tables 都必須改用 cropped_page ---
                tables = cropped_page.find_tables() 
                table_bboxes = [table.bbox for table in tables]
                
                f.write("### Tables\n\n")
                # 這裡也要換成 cropped_page
                extracted_tables = cropped_page.extract_tables() 
                for t in extracted_tables:
                    md_table = table_to_markdown(t)
                    f.write(md_table + "\n")
                    
                f.write("### Text\n\n")
                # 過濾純文字時，一樣要基於 cropped_page
                filtered_page = cropped_page.filter(lambda obj: not_within_bboxes(obj, table_bboxes))
                text = filtered_page.extract_text()
                if text:
                    f.write(text + "\n\n")

    print(f"Extraction completed. Results saved in {output_filename}")
except FileNotFoundError:
     print(f"Error: The file '{pdf_filename}' was not found in the current directory.")
except Exception as e:
     print(f"An unexpected error occurred: {e}")