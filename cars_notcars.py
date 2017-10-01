import glob

cars = glob.glob('vehicles/*/*.png')
notcars = glob.glob('non-vehicles/*/*.png')

print('{} cars, {} notcars'.format(len(cars), len(notcars)))
