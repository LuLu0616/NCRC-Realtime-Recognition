# -*- coding: utf-8 -*-
"""
This module contains classes, functions and an example (main) for handling AER vision data.
"""
import glob
import cv2
import numpy as np
import timer
import matplotlib
import matplotlib.pyplot as plt


class Events(object):
    """
    Temporal Difference events.
    data: a NumPy Record Array with the following named fields
        x: pixel x coordinate, unsigned 16bit int
        y: pixel y coordinate, unsigned 16bit int
        p: polarity value, boolean. False=off, True=on
        ts: timestamp in microseconds, unsigned 64bit int
    width: The width of the frame. Default = 304.
    height: The height of the frame. Default = 240.
    """

    def __init__(self, num_events, width=304, height=240):
        """num_spikes: number of events this instance will initially contain"""
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.uint64)],
                                 shape=(num_events))
        self.width = width
        self.height = height

    def show_td(self, wait_delay=0.01):
        """Displays the TD events (change detection ATIS or DVS events)
        waitDelay: milliseconds
        """
        
        frame_length = wait_delay * pow(10, 6)
        t_max = self.data.ts[-1]
        frame_start = self.data[0].ts
        frame_end = self.data[0].ts + frame_length
        td_img = np.ones((self.height, self.width), dtype=np.uint8)
        plt.ion()
        while frame_start < t_max:
            frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]

            if frame_data.size > 0:
                td_img.fill(128)

                # with timer.Timer() as em_playback_timer:
                for datum in np.nditer(frame_data):
                    td_img[datum['y'].item(0), datum['x'].item(0)] = datum['p'].item(0)
                # print 'prepare td frame by iterating events took %s seconds'
                # %em_playback_timer.secs

                td_img = np.piecewise(td_img, [td_img == 0, td_img == 1, td_img == 128], [0, 255, 128])
                
                plt.subplot(121)
                plt.axis('off')
                plt.title('dvs-input')
                plt.imshow(np.repeat(np.expand_dims(td_img, axis=2), 3, axis=2), cmap='gray')
                plt.subplot(122)
                plt.axis('off')
                plt.title('recognition')
                plt.imshow(cv2.imread('D:\Projects\python\DVS\mnist\\0.png'), 'gray')
                plt.pause(wait_delay)

            frame_start = frame_end + 1
            frame_end = frame_end + frame_length + 1
        return


def read_dataset(filename):
    """Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'"""
    f = open(filename, 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint32(raw_data)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    # Process time stamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    # Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    td = Events(td_indices.size, 34, 34)
    td.data.x = all_x[td_indices]
    td.width = td.data.x.max() + 1
    td.data.y = all_y[td_indices]
    td.height = td.data.y.max() + 1
    td.data.ts = all_ts[td_indices]
    td.data.p = all_p[td_indices]
    return td


def main():
    """Example usage of eventvision"""
    # read in some data
    # td, em = read_aer('0001.val')
    td = read_dataset('D:/Projects/python/DVS/00002.bin')

    # show the TD events
    td.show_td()

if __name__ == "__main__":
    main()
    