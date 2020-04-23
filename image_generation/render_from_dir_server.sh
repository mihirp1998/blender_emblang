# !/bin/bash
for file in /home/sajaved/test/*; do
    file_name="$(basename "$file")"
    file_path="$file/blender_info.p"
    # echo "$file_name"
    /home/sajaved/blender/blender --background --python render_images.py -- --render_from_given_objects 1 --dataset_name $file_name --given_objects_json_path $file_path --all_views 1 --use_gpu 1
done