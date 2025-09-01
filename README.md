# seagrass
dep-seagrass-extents

This flow chart displays the general workflow used for seagrass habitats and wider coastal habitat mapping 

![image](https://github.com/user-attachments/assets/5305a2bc-0ecc-4be7-93a5-2d2faef0865c)


For any issues with json syntax and conflict resolution, I suggest you copy all the code from the files into this https://jsonlint.com/ to validate the json code and then re-enter the code into the notebook files before doing another pull-merge request.

## Testing

Run the command locally or on the DEP Hub to test:

``` bash
python src/run_task.py --tile-id 64,20 --datetime 2024 --version test --output-bucket dep-public-prod
```
