# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL OCTNET AUTHORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import platform

extra_compile_args = ["-ffast-math", '-msse', '-msse2', '-msse3', '-msse4.2']
extra_link_args = []
if 'Linux' in platform.system():
  print('Added OpenMP')
  extra_compile_args.append('-fopenmp')
  extra_link_args.append('-fopenmp')


setup(
  name="cyfusion",
  cmdclass= {'build_ext': build_ext},
  ext_modules=[
    Extension('cyfusion',
      ['cyfusion.pyx'],
      language='c++',
      library_dirs=['./build/'],
      libraries=['m', "fusion_gpu"],
      include_dirs=[np.get_include()],
      extra_compile_args=extra_compile_args,
      extra_link_args=extra_link_args
    )
  ]
)
