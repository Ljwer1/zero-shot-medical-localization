# Data Preparation

This repository does not commit the medical benchmark files to Git. The datasets are large, and redistribution rights may differ from the code license.

The setup follows the `Get Started` workflow of the original `MVFA-AD` project:

1. Optionally follow the `BMAD` instructions to request access to the underlying datasets: https://github.com/DorisBao/BMAD
2. Or download the pre-processed benchmark packages referenced by the original `MVFA-AD` README:
   - Brain: https://drive.google.com/file/d/1YxcjcQqsPdkDO0rqIVHR5IJbqS9EIyoK/view?usp=sharing
   - Liver: https://drive.google.com/file/d/1xriF0uiwrgoPh01N6GlzE5zPi_OIJG1I/view?usp=sharing
   - RESC: https://drive.google.com/file/d/1BqDbK-7OP5fUha5zvS2XIQl-_t8jhTpX/view?usp=sharing
3. Extract the downloaded archives under `data/`.

Expected layout:

```text
data/
|- Brain_AD/
|  |- valid/
|  `- test/
|- Liver_AD/
|  |- valid/
|  `- test/
`- Retina_RESC_AD/
   |- valid/
   `- test/
```

Active protocol in this repository:

- source training split: `valid`
- target validation split during training: `valid`
- target final evaluation split: `test`

`data/` is ignored by Git so local datasets are not uploaded when you publish the repository.
