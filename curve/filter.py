"""
The extraction of the curves from the acv files are from here:
  
  https://github.com/vbalnt/filterizer
"""

from struct import unpack
from scipy import interpolate

from PIL import Image
import numpy, random
import scipy
import pdb
import sys

class rand_Filter:

  def __init__(self):
    self.curves = self._read_curves()
    self.polynomials = self._find_coefficients()
  
  def _random_curve(self, nr_curves):
    curves = []
    for i in range(nr_curves-1):
      curve = [(0,0)]
      # exlcude the 0 and 255
      _x = numpy.sort(random.sample(range(1, 255), 32))
      _y = numpy.sort(random.sample(range(1, 255), 32))
      #_x = numpy.sort(numpy.random.randint(1, 255, 2))
      #_y = numpy.sort(numpy.random.randint(1, 255, 2))
      # _x[0] and _x[1] can't be the same
      
      curve.append((_x[0], _y[0]))
      curve.append((_x[1], _y[1]))
      curve.append((255,255))
      curves.append(curve)
    curves.append([(255,255)])
    return curves

  def _read_curves(self):
    nr_curves = 5
    curves = self._random_curve(nr_curves)
    return curves

  def _find_coefficients(self):
    polynomials = []
    for curve in self.curves:
      xdata = [x[0] for x in curve]
      ydata = [x[1] for x in curve]
      p = interpolate.lagrange(xdata, ydata)
      polynomials.append(p)
    return polynomials

  def get_r(self):
    return self.polynomials[1]

  def get_g(self):
    return self.polynomials[2]

  def get_b(self):
    return self.polynomials[3]

  def get_c(self):
    return self.polynomials[0]

class FilterManager:

  def __init__(self):
    self.filters = None

  def set_filter(self,filter_obj):
    # Overwrites if such a filter already exists
    # NOTE: Fix or not to fix?
    self.filters = filter_obj

  def apply_filter(self, image_array):

    assert self.filters != None
    if image_array.ndim < 3:
      raise Exception('Photos must be in color, meaning at least 3 channels')
    else:
      def interpolate(i_arr, f_arr, p, p_c):
        p_arr = p_c(f_arr)
        return p_arr 

      # NOTE: Assumes that image_array is a numpy array
      image_filter = self.filters
      # NOTE: What happens if filter does not exist?
      width,height,channels = image_array.shape
      filter_array = numpy.zeros((width, height, 3), dtype=float) 

      p_r = image_filter.get_r()
      p_g = image_filter.get_g()
      p_b = image_filter.get_b()
      p_c = image_filter.get_c()

      filter_array[:,:,0] = p_r(image_array[:,:,0])
      filter_array[:,:,1] = p_g(image_array[:,:,1])
      filter_array[:,:,2] = p_b(image_array[:,:,2])
      filter_array = filter_array.clip(0,255)
      filter_array = p_c(filter_array)

      filter_array = numpy.ceil(filter_array).clip(0,255)

      return filter_array.astype(numpy.uint8)

if __name__ == '__main__':

  if len(sys.argv) < 3:
    print "Wrong number of arguments"
    print """  Usage: \
          python filter.py [imagefile] [name]"""
  else:
    img_filter = rand_Filter()
    
    im = Image.open(sys.argv[1])

    #im.show()

    image_array = numpy.array(im)

    filter_manager = FilterManager()
    filter_manager.set_filter(img_filter)

    filter_array = filter_manager.apply_filter(image_array)
    im = Image.fromarray(filter_array)
    im.save(sys.argv[2]+'.png')
    #im.show() 
