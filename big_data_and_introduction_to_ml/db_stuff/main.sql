SELECT MEDIAN(SALARY) FROM ROSSTAT_SALARY_RU
WHERE REGION_NAME != 'Смоленская область' AND
REGION_NAME != 'Чувашская Республика' AND
REGION_NAME != 'Костромская область' AND
REGION_NAME != 'Саратовская область'
ORDER BY SALARY ASC;