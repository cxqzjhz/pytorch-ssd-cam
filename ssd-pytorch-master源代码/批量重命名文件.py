import os

print("当前路径: ",os.getcwd())
print("批量重命名文件.....")
image_path = r'G:\Iris_SSD\ssd-pytorch-master\VOCdevkit\VOC2007\Annotations'
for filename in os.listdir(image_path):
    newName = str(filename)
    newName = newName.replace(' ', '')
    os.rename(os.path.join(image_path,filename),os.path.join(image_path,newName))
    print("文件： ",filename,"--->",newName," 重命名已完成！") 
