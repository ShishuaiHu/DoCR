from setuptools import setup, find_namespace_packages

setup(name='docr',
      packages=find_namespace_packages(include=["docr", "docr.*"]),
      version='0.0.1',
      description='Domain specific Convolution and high frequency Reconstruction (DoCR) '
                  'based UDA for medical image segmentation.',
      author='anonymous',
      author_email='anonymous',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
            "torch>=1.6.0a",
            "tqdm",
            "dicom2nifti",
            "scikit-image>=0.14",
            "medpy",
            "scipy",
            "batchgenerators>=0.21",
            "numpy",
            "sklearn",
            "SimpleITK",
            "pandas",
            "requests",
            "hiddenlayer", "graphviz", "IPython",
            "nibabel", 'tifffile',
            "tensorboard"
      ],
      entry_points={
          'console_scripts': [
            'DoCR_train = docr.training.run_training:main',
            'DoCR_test = docr.inference.run_inference:main',
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'unsupervised domain adaptation']
      )
