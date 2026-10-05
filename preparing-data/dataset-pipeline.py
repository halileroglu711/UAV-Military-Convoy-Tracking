#Since there are unwanted classes in the dataset's raw form, we are going to drop them.
import os
sets=[r"./data/test",
      r"./data/train",
      r"./data/valid"]
id_map={
    "2":"0", 
    "3":"1", 
    "4":"2",
    "5":"3",
    "6":"4",
    "7":"1" 
}
drop_ids=["0","1"] #drop artilerry and car

for set in sets:
    labels_dir=set+"/labels"
    images_dir=set+"/images"
    
    for filename in os.listdir(labels_dir):
        imagename=filename.replace(".txt",".jpg")
        label_path=os.path.join(labels_dir,filename)
        image_path=os.path.join(images_dir,imagename)

        with open(label_path,"r") as file:
            lines=file.readlines()
        new_lines=[]
        for line in lines:
            sections=line.strip().split()
            if not sections:
                continue
            class_id=sections[0]
            
            if class_id in drop_ids:
                continue
            elif class_id in id_map:
                sections[0]=id_map[class_id]
                new_lines.append(" ".join(sections)+"\n")
        if len(new_lines)==0:
            os.remove(label_path)
            if os.path.exists(image_path):
                os.remove(image_path)
        else:
            with open(label_path,"w") as file:
                file.writelines(new_lines)

print("Pipeline has ended. Dataset was cleared and unwanted classes are gone.")