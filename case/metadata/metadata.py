from PIL import Image
from PIL.ExifTags import TAGS
import piexif
import pandas as pd

def extract_metadata(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Extract standard metadata
    metadata = {}
    try:
        exifdata = image.getexif()
        if exifdata:
            for tag_id in exifdata:
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                if isinstance(data, bytes):
                    data = data.decode(errors='replace')
                metadata[tag] = data
    except Exception as e:
        metadata['EXIF Error'] = str(e)
    
    # Extract EXIF using piexif
    try:
        exif_dict = piexif.load(image.info['exif'])
        for ifd in exif_dict:
            if ifd != 'thumbnail':
                for tag in exif_dict[ifd]:
                    tag_name = piexif.TAGS[ifd][tag]["name"]
                    tag_value = exif_dict[ifd][tag]
                    if isinstance(tag_value, bytes):
                        tag_value = tag_value.decode(errors='replace')
                    metadata[tag_name] = tag_value
    except:
        pass
    
    # Check for inconsistencies
    inconsistencies = []
    if 'Software' in metadata and 'Photoshop' in metadata['Software']:
        inconsistencies.append('Document may have been edited with Photoshop')
    if 'ModifyDate' in metadata and 'CreateDate' in metadata:
        if metadata['ModifyDate'] != metadata['CreateDate']:
            inconsistencies.append('Create and modify dates differ')
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(list(metadata.items()), columns=['Tag', 'Value'])
    
    return df, inconsistencies

# Usage
metadata_df, inconsistencies = extract_metadata('document.jpg')
print(metadata_df)
print("\nPotential Inconsistencies:", inconsistencies)