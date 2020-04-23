# !/bin/bash
for file in /Users/ashar/Downloads/test/*; do
    file_name="$(basename "$file")"
    file_path="$file/blender_info.json"
    # echo "$file_name"
    /Applications/blender/blender.app/Contents/MacOS/blender --background --python render_images.py -- --render_from_given_objects 1 --dataset_name $file_name --given_objects_json_path $file_path --all_views 1
done