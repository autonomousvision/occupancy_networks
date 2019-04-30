import os
import common
import argparse
import numpy as np
from multiprocessing import Pool


class Scale:
    """
    Scales a bunch of meshes.
    """

    def __init__(self):
        """
        Constructor.
        """

        parser = self.get_parser()
        self.options = parser.parse_args()

    def get_parser(self):
        """
        Get parser of tool.

        :return: parser
        """

        parser = argparse.ArgumentParser(description='Scale a set of meshes stored as OFF files.')
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument('--in_dir', type=str,
                                 help='Path to input directory.')
        input_group.add_argument('--in_file', type=str,
                                 help='Path to input directory.')
        parser.add_argument('--out_dir', type=str, required=True,
                            help='Path to output directory; files within are overwritten!')
        parser.add_argument('--t_dir', type=str,
                            help='Path to transformation directory; files within are overwritten!')
        parser.add_argument('--padding', type=float, default=0.1,
                            help='Padding applied to the sides (in total).')

        parser.add_argument('--n_proc', type=int, default=0,
                            help='Number of processes to run in parallel'
                                 '(0 means sequential execution).')
        parser.add_argument('--overwrite', action='store_true',
                            help='Overwrites existing files if true.')
        return parser

    def read_directory(self, directory):
        """
        Read directory.

        :param directory: path to directory
        :return: list of files
        """

        files = []
        for filename in os.listdir(directory):
            files.append(os.path.normpath(os.path.join(directory, filename)))

        return files

    def get_in_files(self):
        if self.options.in_dir is not None:
            assert os.path.exists(self.options.in_dir)
            common.makedir(self.options.out_dir)
            files = self.read_directory(self.options.in_dir)
        else:
            files = [self.options.in_file]

        if not self.options.overwrite:
            def file_filter(filepath):
                outpath_m, outpath_t = self.get_outpath(filepath)
                file_exists = os.path.exists(outpath_m)
                if outpath_t is not None:
                    file_exists = (file_exists and os.path.exists(outpath_t))
                return not file_exists
            files = list(filter(file_filter, files))

        return files



    def get_outpath(self, filepath):
        filename_m = os.path.basename(filepath)
        outpath_m = os.path.join(self.options.out_dir, filename_m)

        if self.options.t_dir is not None:
            filename_t =  os.path.splitext(filename_m)[0] + '.npz'
            outpath_t = os.path.join(self.options.t_dir, filename_t)
        else:
            outpath_t = None

        return outpath_m, outpath_t


    def run(self):
        """
        Run the tool, i.e. scale all found OFF files.
        """
        common.makedir(self.options.out_dir)
        if self.options.t_dir is not None:
            common.makedir(self.options.t_dir)

        files = self.get_in_files()

        if self.options.n_proc == 0:
            for filepath in files:
                self.run_file(filepath)
        else:
            with Pool(self.options.n_proc) as p:
                p.map(self.run_file, files)

    def run_file(self, filepath):
        mesh = common.Mesh.from_off(filepath)

        # Get extents of model.
        bb_min, bb_max = mesh.extents()
        bb_min, bb_max = np.array(bb_min), np.array(bb_max)
        total_size = (bb_max - bb_min).max()

        # Set the center (although this should usually be the origin already).
        centers = (
            (bb_min[0] + bb_max[0]) / 2,
            (bb_min[1] + bb_max[1]) / 2,
            (bb_min[2] + bb_max[2]) / 2
        )
        # Scales all dimensions equally.
        scale = total_size / (1 - self.options.padding)

        translation = (
            -centers[0],
            -centers[1],
            -centers[2]
        )
        scales_inv = (
            1/scale, 1/scale, 1/scale
        )

        mesh.translate(translation)
        mesh.scale(scales_inv)

        print('[Data] %s extents before %f - %f, %f - %f, %f - %f'
            % (os.path.basename(filepath),
                bb_min[0], bb_max[0], bb_min[1], bb_max[1], bb_min[2], bb_max[2]))
        bb1_min, bb1_max = mesh.extents()
        print('[Data] %s extents after %f - %f, %f - %f, %f - %f'
            % (os.path.basename(filepath),
                bb1_min[0], bb1_max[0], bb1_min[1], bb1_max[1], bb1_min[2], bb1_max[2]))

        # May also switch axes if necessary.
        # mesh.switch_axes(1, 2)

        outpath_m, outpath_t = self.get_outpath(filepath)

        mesh.to_off(outpath_m)

        if outpath_t is not None:
            np.savez(outpath_t,
                     loc=centers, scale=scale,
                     bb0_min=bb_min, bb0_max=bb_max,
                     bb1_min=bb1_min, bb1_max=bb1_max)

if __name__ == '__main__':
    app = Scale()
    app.run()
