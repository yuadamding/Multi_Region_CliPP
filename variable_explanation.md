# The Basic Classes
## snv
snv contains all of the information in the observations with respect to a single snv. The attributes of snv are 
| Attributes | Type| Description |
| ----------- | ----------- | ----------- |
| name | str| name or id of this snv|
|num_regions| int | number of sampeles that contains this snv|
|reads | np.array | an array contains reads of this snv in each region
|total_reads| np.array| an array contains total number of reads in each region
|rho | np.array| an array contains tumor purities in each region|
|tumor_cn | np.array| an array contains tumor cell copy number in each region|
|normal_cn | np.array| an array contains normal cell copy number in each region|
|major_cn | np.array| an array contains tumor cell major allele  copy number in each region|
|minor_cn | np.array| an array contains tumor cell minor allele  copy number in each region|
|cp | np.array| an array contains celluar prevalence in each region|
|prop | np.array| an array contains expected proportion in each region, a parameter of Binomial distribution|
|specific_copy_number | np.array| an array contains  SNV-specific copy number in each region|
|likelihood | np.array | an array contains  the corresponding loglikelihood function value in each region|
|mapped_cp | np.array | convert cp into the whole real space to avoid box constraint|
|map_method | str| the method to convert cp, normally should be the inverse of a cdf function

## snvs
snvs contains all snv objects.
| Attributes | Type| Description |
| ----------- | ----------- | ----------- |
| num_snvs | int | number of snvs in contained in this object |
| num_regions | int | number of regions each snv getting observed |
| likelihood | float | total loglikelihood value, equals to the sum of each snv's  loglikelihood|
| snv_lst | list | a list of snv objects, usually contains the raw info |
| p | np.array  | vector, the vectorized P matrix, dim: num_snvs*num_regions  |
| combination | list | a list of set, such sets are total possible paris of two snvs |
| v | np.array | vector, dim: len(combination)*num_regions |
| y | np.array | vector, dim: len(combination)*num_regions |
| gamma | float | a parameter for augmented lagrangian method |
| omega | np.array | vector,  dim: len(combination)|
| paris_mapping | dictionary | mapping from combination to index, used to order v and y |
| paris_mapping_inverse | dictionary | inverse of paris_mapping, used to find the corresponding pari given index of v and y |
