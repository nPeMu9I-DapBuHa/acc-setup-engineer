""" Parser for MoTec ld files
Code created through reverse engineering the data format.
"""
import os, glob, copy


import datetime
import struct

import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET


class ldData(object):
    """Container for parsed data of an ld file.
    Allows reading and writing.
    """

    def __init__(self, head, channs):
        self.head = head
        self.channs = channs

    #unused
    
    def __getitem__(self, item):
        if not isinstance(item, int):
            col = [n for n, x in enumerate(self.channs) if x.name == item]
            if len(col) != 1:
                raise Exception("Could get column", item, col)
            item = col[0]
        return self.channs[item]

    def __iter__(self):
        return iter([x.name for x in self.channs])

    @classmethod
    def frompd(cls, df):
        # for now, fix datatype and frequency
        freq, dtype = 10, np.float32

        # pointer to meta data of first channel
        meta_ptr = struct.calcsize(ldHead.fmt)

        # list of columns to read - only accept numeric data
        cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

        # pointer to data of first channel
        chanheadsize = struct.calcsize(ldChan.fmt)
        data_ptr = meta_ptr + len(cols) * chanheadsize

        # create a mocked header
        head = ldHead(meta_ptr, data_ptr, 0,  None,
                       "testdriver",  "testvehicleid", "testvenue",
                       datetime.datetime.now(),
                       "just a test", "testevent", "practice")

        # create the channels, meta data and associated data
        channs, prev, next = [], 0, meta_ptr + chanheadsize
        for n, col in enumerate(cols):
            # create mocked channel header
            chan = ldChan(None,
                          meta_ptr, prev, next if n < len(cols)-1 else 0,
                          data_ptr, len(df[col]),
                          dtype, freq, 0, 1, 1, 0,
                          col, col, "m")

            # link data to the channel
            chan._data = df[col].to_numpy(dtype)

            # calculate pointers to the previous/next channel meta data
            prev = meta_ptr
            meta_ptr = next
            next += chanheadsize

            # increment data pointer for next channel
            data_ptr += chan._data.nbytes

            channs.append(chan)

        return cls(head, channs)

    @classmethod
    def fromfile(cls, f):
        # type: (str) -> ldData
        """Parse data of an ld file
        """
        return cls(*read_ldfile(f))

    def write(self, f):
        # type: (str) -> ()
        """Write an ld file containing the current header information and channel data
        """

        # convert the data using scale/shift etc before writing the data
        conv_data = lambda c: ((c.data / c.mul) - c.shift) * c.scale / pow(10., -c.dec)

        with open(f, 'wb') as f_:
            self.head.write(f_, len(self.channs))
            f_.seek(self.channs[0].meta_ptr)
            list(map(lambda c: c[1].write(f_, c[0]), enumerate(self.channs)))
            list(map(lambda c: f_.write(conv_data(c)), self.channs))


class ldEvent(object):
    fmt = '<64s64s1024sH'

    def __init__(self, name, session, comment, venue_ptr, venue):
        self.name, self.session, self.comment, self.venue_ptr, self.venue = \
            name, session, comment, venue_ptr, venue

    @classmethod
    def fromfile(cls, f):
        # type: (file) -> ldEvent
        """Parses and stores the event information in an ld file
        """
        name, session, comment, venue_ptr = struct.unpack(
            ldEvent.fmt, f.read(struct.calcsize(ldEvent.fmt)))
        name, session, comment = map(decode_string, [name, session, comment])

        venue = None
        if venue_ptr > 0:
            f.seek(venue_ptr)
            venue = ldVenue.fromfile(f)

        return cls(name, session, comment, venue_ptr, venue)

    def write(self, f):
        f.write(struct.pack(ldEvent.fmt,
                            self.name.encode(),
                            self.session.encode(),
                            self.comment.encode(),
                            self.venue_ptr))

        if self.venue_ptr > 0:
            f.seek(self.venue_ptr)
            self.venue.write(f)

    def __str__(self):
        return "%s; venue: %s"%(self.name, self.venue)


class ldVenue(object):
    fmt = '<64s1034xH'

    def __init__(self, name, vehicle_ptr, vehicle):
        self.name, self.vehicle_ptr, self.vehicle = name, vehicle_ptr, vehicle

    @classmethod
    def fromfile(cls, f):
        # type: (file) -> ldVenue
        """Parses and stores the venue information in an ld file
        """
        name, vehicle_ptr = struct.unpack(ldVenue.fmt, f.read(struct.calcsize(ldVenue.fmt)))

        vehicle = None
        if vehicle_ptr > 0:
            f.seek(vehicle_ptr)
            vehicle = ldVehicle.fromfile(f)
        return cls(decode_string(name), vehicle_ptr, vehicle)

    def write(self, f):
        f.write(struct.pack(ldVenue.fmt, self.name.encode(), self.vehicle_ptr))

        if self.vehicle_ptr > 0:
            f.seek(self.vehicle_ptr)
            self.vehicle.write(f)

    def __str__(self):
        return "%s; vehicle: %s"%(self.name, self.vehicle)


class ldVehicle(object):
    fmt = '<64s128xI32s32s'

    def __init__(self, id, weight, type, comment):
        self.id, self.weight, self.type, self.comment = id, weight, type, comment

    @classmethod
    def fromfile(cls, f):
        # type: (file) -> ldVehicle
        """Parses and stores the vehicle information in an ld file
        """
        id, weight, type, comment = struct.unpack(ldVehicle.fmt, f.read(struct.calcsize(ldVehicle.fmt)))
        id, type, comment = map(decode_string, [id, type, comment])
        return cls(id, weight, type, comment)

    def write(self, f):
        f.write(struct.pack(ldVehicle.fmt, self.id.encode(), self.weight, self.type.encode(), self.comment.encode()))

    def __str__(self):
        return "%s (type: %s, weight: %i, %s)"%(self.id, self.type, self.weight, self.comment)


class ldHead(object):
    fmt = '<' + (
        "I4x"     # ldmarker
        "II"      # chann_meta_ptr chann_data_ptr
        "20x"     # ??
        "I"       # event_ptr
        "24x"     # ??
        "HHH"     # unknown static (?) numbers
        "I"       # device serial
        "8s"      # device type
        "H"       # device version
        "H"       # unknown static (?) number
        "I"       # num_channs
        "4x"      # ??
        "16s"     # date
        "16x"     # ??
        "16s"     # time
        "16x"     # ??
        "64s"     # driver
        "64s"     # vehicleid
        "64x"     # ??
        "64s"     # venue
        "64x"     # ??
        "1024x"   # ??
        "I"       # enable "pro logging" (some magic number?)
        "66x"     # ??
        "64s"     # short comment
        "126x"    # ??
        "64s"     # event
        "64s"     # session
    )

    def __init__(self, meta_ptr, data_ptr, aux_ptr, aux, driver, vehicleid, venue, datetime, short_comment, event, session):
        self.meta_ptr, self.data_ptr, self.aux_ptr, self.aux, self.driver, self.vehicleid, \
        self.venue, self.datetime, self.short_comment, self.event, self.session = meta_ptr, data_ptr, aux_ptr, aux, \
                                                driver, vehicleid, venue, datetime, short_comment, event, session

    @classmethod
    def fromfile(cls, f):
        # type: (file) -> ldHead
        """Parses and stores the header information of an ld file
        """
        (_, meta_ptr, data_ptr, aux_ptr,
            _, _, _,
            _, _, _, _, n,
            date, time,
            driver, vehicleid, venue,
            _, short_comment, event, session) = struct.unpack(ldHead.fmt, f.read(struct.calcsize(ldHead.fmt)))
        date, time, driver, vehicleid, venue, short_comment, event, session = \
            map(decode_string, [date, time, driver, vehicleid, venue, short_comment, event, session])

        try:
            # first, try to decode datatime with seconds
            _datetime = datetime.datetime.strptime(
                    '%s %s'%(date, time), '%d/%m/%Y %H:%M:%S')
        except ValueError:
            _datetime = datetime.datetime.strptime(
                '%s %s'%(date, time), '%d/%m/%Y %H:%M')

        aux = None
        if aux_ptr > 0:
            f.seek(aux_ptr)
            aux = ldEvent.fromfile(f)
        return cls(meta_ptr, data_ptr, aux_ptr, aux, driver, vehicleid, venue, _datetime, short_comment, event, session)

    def write(self, f, n):
        f.write(struct.pack(ldHead.fmt,
                            0x40,
                            self.meta_ptr, self.data_ptr, self.aux_ptr,
                            1, 0x4240, 0xf,
                            0x1f44, "ADL".encode(), 420, 0xadb0, n,
                            self.datetime.date().strftime("%d/%m/%Y").encode(),
                            self.datetime.time().strftime("%H:%M:%S").encode(),
                            self.driver.encode(), self.vehicleid.encode(), self.venue.encode(),
                            0xc81a4, self.short_comment.encode(), self.event.encode(), self.session.encode(),
                            ))
        if self.aux_ptr > 0:
            f.seek(self.aux_ptr)
            self.aux.write(f)

    def __str__(self):
        return 'driver:    %s\n' \
               'vehicleid: %s\n' \
               'venue:     %s\n' \
               'event:     %s\n' \
               'session:   %s\n' \
               'short_comment: %s\n' \
               'event_long:    %s'%(
            self.driver, self.vehicleid, self.venue, self.event, self.session, self.short_comment, self.aux)


class ldChan(object):
    """Channel (meta) data
    Parses and stores the channel meta data of a channel in a ld file.
    Needs the pointer to the channel meta block in the ld file.
    The actual data is read on demand using the 'data' property.
    """

    fmt = '<' + (
        "IIII"    # prev_addr next_addr data_ptr n_data
        "H"       # some counter?
        "HHH"     # datatype datatype rec_freq
        "HHHh"    # shift mul scale dec_places
        "32s"     # name
        "8s"      # short name
        "12s"     # unit
        "40x"     # ? (40 bytes for ACC, 32 bytes for acti)
    )

    def __init__(self, _f, meta_ptr, prev_meta_ptr, next_meta_ptr, data_ptr, data_len,
                 dtype, freq, shift, mul, scale, dec,
                 name, short_name, unit):

        self._f = _f
        self.meta_ptr = meta_ptr
        self._data = None

        (self.prev_meta_ptr, self.next_meta_ptr, self.data_ptr, self.data_len,
        self.dtype, self.freq,
        self.shift, self.mul, self.scale, self.dec,
        self.name, self.short_name, self.unit) = prev_meta_ptr, next_meta_ptr, data_ptr, data_len,\
                                                 dtype, freq,\
                                                 shift, mul, scale, dec,\
                                                 name, short_name, unit

    @classmethod
    def fromfile(cls, _f, meta_ptr):
        # type: (str, int) -> ldChan
        """Parses and stores the header information of an ld channel in a ld file
        """
        with open(_f, 'rb') as f:
            f.seek(meta_ptr)

            (prev_meta_ptr, next_meta_ptr, data_ptr, data_len, _,
             dtype_a, dtype, freq, shift, mul, scale, dec,
             name, short_name, unit) = struct.unpack(ldChan.fmt, f.read(struct.calcsize(ldChan.fmt)))

        name, short_name, unit = map(decode_string, [name, short_name, unit])

        if dtype_a in [0x07]:
            dtype = [None, np.float16, None, np.float32][dtype-1]
        elif dtype_a in [0, 0x03, 0x05]:
            dtype = [None, np.int16, None, np.int32][dtype-1]
        else: raise Exception('Datatype %i not recognized'%dtype_a)

        return cls(_f, meta_ptr, prev_meta_ptr, next_meta_ptr, data_ptr, data_len,
                   dtype, freq, shift, mul, scale, dec,name, short_name, unit)

    def write(self, f, n):
        if self.dtype == np.float16 or self.dtype == np.float32:
            dtype_a = 0x07
            dtype = {np.float16: 2, np.float32: 4}[self.dtype]
        else:
            dtype_a = 0x03
            dtype = {np.int16: 2, np.int32: 4}[self.dtype]

        f.write(struct.pack(ldChan.fmt,
                            self.prev_meta_ptr, self.next_meta_ptr, self.data_ptr, self.data_len,
                            0x2ee1+n, dtype_a, dtype, self.freq, self.shift, self.mul, self.scale, self.dec,
                            self.name.encode(), self.short_name.encode(), self.unit.encode()))

    @property
    def data(self):
        # type: () -> np.array
        """ Read the data words of the channel
        """
        if self._data is None:
            # jump to data and read
            with open(self._f, 'rb') as f:
                f.seek(self.data_ptr)
                try:
                    self._data = np.fromfile(f,
                                            count=self.data_len, dtype=self.dtype)

                    self._data = (self._data/self.scale * pow(10., -self.dec) + self.shift) * self.mul

                    if len(self._data) != self.data_len:
                        raise ValueError("Not all data read!")

                except ValueError as v:
                    print(v, self.name, self.freq,
                          hex(self.data_ptr), hex(self.data_len),
                          hex(len(self._data)),hex(f.tell()))
                    # raise v
        return self._data

    def __str__(self):
        return 'chan %s (%s) [%s], %i Hz'%(
            self.name,
            self.short_name, self.unit,
            self.freq)


def decode_string(bytes):
    # type: (bytes) -> str
    """decode the bytes and remove trailing zeros
    """
    try:
        return bytes.decode('ascii').strip().rstrip('\0').strip()
    except Exception as e:
        print("Could not decode string: %s - %s"%(e, bytes))
        return ""
        # raise e

def read_channels(f_, meta_ptr):
    # type: (str, int) -> list
    """ Read channel data inside ld file
    Cycles through the channels inside an ld file,
     starting with the one where meta_ptr points to.
     Returns a list of ldchan objects.
    """
    chans = []
    while meta_ptr:
        chan_ = ldChan.fromfile(f_, meta_ptr)
        chans.append(chan_)
        meta_ptr = chan_.next_meta_ptr
    return chans


def read_ldfile(f_):
    # type: (str) -> (ldHead, list)
    """ Read an ld file, return header and list of channels
    """
    head_ = ldHead.fromfile(open(f_,'rb'))
    chans = read_channels(f_, head_.meta_ptr)
    return head_, chans


def laps(f):
    laps = []
    print(os.path.splitext(f)[0])
    tree = ET.parse(os.path.splitext(f)[0]+".ldx")
    root = tree.getroot()

    # read lap times
    for lap in root[0][0][0][0]:
        laps.append(float(lap.attrib['Time'])*1e-6)
    return laps


def laps_limits(laps, freq, n):
    """find the start/end indizes of the data for each lap
    """
    laps_limits = []
    if laps[0]!=0:
        laps_limits = [0]
    laps_limits.extend((np.array(laps)*freq).astype(int))
    laps_limits.extend([n])
    return list(zip(laps_limits[:-1], laps_limits[1:]))


def laps_times(laps):
    """calculate the laptime for each lap"""
    laps_times = []
    if len(laps) == 0: return laps_times
    if laps[0] != 0:  laps_times = [laps[0]]
    laps_times.extend(list(laps[1:]-laps[:-1]))
    return laps_times

def laps_timedeltas(laps):
    """calculate the laptime for each lap"""
    return [datetime.timedelta(seconds=s) for s in laps_times(laps)]


class DataStore(object):
    @staticmethod
    def create_track(df, laps_times=None):
        # dx = (2*r*np.tan(alpha/2)) * np.cos(heading)
        # dy = (2*r*np.tan(alpha/2)) * np.sin(heading)
        dx = df.ds * np.cos(df.heading)
        dy = df.ds * np.sin(df.heading)

        # calculate correction to close the track
        # use best lap
        if laps_times is None:
            df_ = df
        else:
            fastest = np.argmin([999999 if x==0 else x for x in laps_times])
            df_ = df[(df.lap==fastest)]
        fac = 1.
        dist = None
        n = 0
        while n < 1000:
            dx = df_.ds * np.cos(df_.heading*fac)
            dy = df_.ds * np.sin(df_.heading*fac)
            end = (dx.cumsum()).values[-1], (dy.cumsum()).values[-1]
            # print(end, dist, fac)

            newdist = np.sqrt(end[0]**2+end[1]**2)
            if dist is not None and newdist>dist: break
            dist = newdist
            fac -= 0.0001
            n += 1

        if n == 1000:
            fac = 1.

        # recalculate with correction
        df.alpha = df.alpha*fac
        df.heading = df.alpha.cumsum()
        dx = df.ds * np.cos(df.heading*fac)
        dy = df.ds * np.sin(df.heading*fac)
        x = dx.cumsum()
        y = dy.cumsum()

        df = pd.concat([df, pd.DataFrame(
            {'x':x,'y':y,
             'dx':dx, 'dy':dy,
             })], axis=1)

        return df

    @staticmethod
    def calc_over_understeer(df):
        # calculate oversteer, based on math in ACC MoTec workspace
        wheelbase = 2.645
        neutral_steering = (wheelbase * df.alpha * 180/np.pi).rolling(10).mean()

        steering_corr= (df.steerangle/11)
        oversteer  = np.sign(df.g_lat) * (neutral_steering-steering_corr)
        understeer = oversteer.copy()

        indices = understeer > 0
        understeer[indices] = 0

        df = pd.concat([df, pd.DataFrame(
            {'steering_corr':steering_corr,
             'neutral_steering':neutral_steering,
             'oversteer':oversteer,
             'understeer':understeer})], axis=1)
        return df

    @staticmethod
    def add_cols(df, freq=None, n=None, laps_limits=None, lap=None):
        if 'speedkmh' not in df.columns:
            df = pd.concat([df, pd.DataFrame({'speedkmh': df.speed*3.6})], axis=1)
        if 'speed' not in df.columns:
            df = pd.concat([df, pd.DataFrame({'speed': df.speedkmh/3.6})], axis=1)

        # create list with the distance
        if 'ds' in df.columns:
            ds = df.ds
        else:
            ds = (df.speed / freq)
            df = pd.concat([df, pd.DataFrame({'ds': ds})], axis=1)

        # create list with total time
        if 'dt' in df.columns:
            t = df.dt.cumsum()
        else:
            t = np.arange(n)*(1/freq)

        # create list with the lap number, distance in lap, time in lap
        s = np.array(df.ds.cumsum())
        if laps_limits is None:
            l, sl, tl = [lap]*len(s), s, t
        else:
            l, sl, tl = [], [], []
            for n, (n1, n2) in enumerate(laps_limits):
                l.extend([n]*(n2-n1))
                sl.extend(list(s[n1:n2]-s[n1]))
                tl.extend(list(t[n1:n2]-t[n1]))

        # for calculate of x/y position on track from speed and g_lat
        gN = 9.81
        r = 1 / (gN * df.g_lat/df.speed.pow(2))
        alpha = ds / r
        heading = alpha.cumsum()

        # add the lists to the dataframe
        df = pd.concat([df, pd.DataFrame(
            {'lap':l,
             'g_sum': df.g_lon.abs()+df.g_lat.abs(),
             'heading':heading,
             'alpha':alpha,
             'dist':s,'dist_lap':sl,
             'time':t,'time_lap':tl})], axis=1)

        return df

    def get_data_frame(self, lap=None):
        pass

class TelemetryChann():
    def __init__(self, data, freq, lap_limits):
        self.data = data
        self.freq = freq
        self.laps_limits = lap_limits

    @classmethod
    def from_telemetry_chan(cls, telemetry_chan, data_scale_factor=1):
        data = telemetry_chan.data
        freq = telemetry_chan.freq
        laps_limits = telemetry_chan.laps_limits

        return cls(data * data_scale_factor, freq, laps_limits)

    @classmethod
    def from_data(cls, data, freq, laps_limits):
        return cls(data, freq, laps_limits)

    @classmethod
    def from_chan(cls, chan, laps):
        data = chan.data
        freq = chan.freq
        lap_limits = laps_limits(laps, freq, len(data))
        return cls(data, freq, lap_limits)

    @classmethod
    def from_channs(cls, channs, laps):
        channs_dict = {}
        for chan in channs:
            data = chan.data
            freq = chan.freq
            lap_limits = laps_limits(laps, freq, len(data))
            channs_dict[chan.name.lower()] = cls(data, freq, lap_limits)
        return channs_dict

    def round(self, resolution=0):
        return TelemetryChann(np.around(self.data, decimals=resolution), self.freq, self.laps_limits)

    def get_data(self, lap_stint=None):
        if lap_stint == None:
            return np.copy(self.data)
        return np.copy(self.data[self.laps_limits[lap_stint[0]][0]:self.laps_limits[lap_stint[0] + lap_stint[1]][0] + 1])


class DataChan:
    def __init__(self, data, freq, unit):
        self.data = data
        self.freq = freq
        self.unit = unit

class MyLDDataStore(DataStore):
    def __init__(self, channs, laps, steer_ratio):
        print(laps)
        self.laps = laps
        for c in channs:
           print(c.name, c.freq, c.unit)
        self.channs = { self.chan_name(c) : DataChan(c.data, c.freq, c.unit) for c in channs }
        if 'speedkmh' not in self.channs:
            self.channs['speedkmh'] = DataChan(self.channs["speed"].data * 3.6, self.channs["speed"].freq, "km/h")
        if 'speed' not in self.channs:
            self.channs['speed'] = DataChan(self.channs["speedkmh"].data / 3.6, self.channs["speedkmh"].freq, "m/s")

        freq =  self.channs['speed'].freq
        if 'ds' in self.channs:
            ds = self.channs['ds'].data
        else:
            ds = (self.channs['speed'].data / freq)


        s = np.array(ds.cumsum())
        t = np.arange(0, len(s)+2 / freq, 1 / freq)
        t = t[:len(s)]
        lap_limits = laps_limits(self.laps, freq, len(s))
        sl, tl = [], []
        for n1, n2 in lap_limits:
            sl.extend(list(s[n1:n2]-s[n1]))
            tl.extend(list(t[n1:n2]-t[n1]))

        self.meta_map = {freq : {"dist": np.array(s), "lap_dist": np.array(sl), "time": np.array(t), "lap_time": np.array(tl), "lap_limits": np.array(lap_limits)}}

        channels = ["g_lat", "speed"]
        g_lat = self.channs["g_lat"].data
        speed = self.channs["speed"].data[::3]
        ds = ds[::3]

        # check here

        if speed.size != g_lat.size:
            speed = speed[:min(g_lat.size, speed.size)]
            ds = ds[:min(g_lat.size, speed.size)]
            g_lat = g_lat[:min(g_lat.size, speed.size)]


        for k, v in self.channs.items():
            freq = v.freq
            if freq not in self.meta_map:
                time = np.arange(0, len(v.data) / freq, 1 / freq)
                dist = np.interp(time, t, s)
                sl, tl = [], []
                lap_limits = laps_limits(self.laps, freq, len(dist))
                for n1, n2 in lap_limits:
                    sl.extend(list(dist[n1:n2]-dist[n1]))
                    tl.extend(list(time[n1:n2]-time[n1]))
                self.meta_map[freq] = {"dist": np.array(dist), "lap_dist": np.array(sl), "time": np.array(time), "lap_time": np.array(tl), "lap_limits": np.array(lap_limits)}

        self.channs["steerangle"].data = self.channs["steerangle"].data / steer_ratio
        gN = 9.81
        r = 1 / (gN * g_lat/np.square(speed))
        alpha = ds / r
        wheelbase = 2.6416
        neutral_steering = (wheelbase * alpha * 180/np.pi)

        steering_corr= (self.channs["steerangle"].data)[::3]
        oversteer  = np.sign(g_lat) * (neutral_steering-steering_corr)

        oversteer2 = ((steering_corr * np.square(speed) * np.pi/180) / (wheelbase * gN * g_lat) - 1) / np.square(speed)

        self.channs["oversteer"]  = DataChan( oversteer, 20, "deg")
        self.channs["oversteer2"]  = DataChan( oversteer2, 20, "absolute")
        self.channs["steerangle_calc"]  = DataChan( neutral_steering, 20, "deg")

        self.laps_times = laps_times(laps)
        self.laps_timedeltas = laps_timedeltas(laps)
        self.unit_map = {}
        for name, c in self.channs.items():
            if c.unit in self.unit_map:
                self.unit_map[c.unit].append(name)
            else:
                self.unit_map[c.unit] = [name]    
        

    def get_unit_map(self):
        return self.unit_map

    def get_units(self):
        return list(self.unit_map.keys())

    def chan_name(self, x):
        return x.name.lower()

    def get_chan_names(self):
        return sorted(list(self.channs.keys()))

    def get_chan_unit(self, chan_name):
        return self.channs[chan_name].unit

    def chan_data(self, chan_name, lap_stint=None):
        if lap_stint == None or lap_stint[0] == None:
            return np.copy(self.channs[chan_name].data)
        l_limits = self.meta_map[self.channs[chan_name].freq]['lap_limits']
        return np.copy(self.channs[chan_name].data[l_limits[lap_stint[0]][0]:l_limits[lap_stint[0] + lap_stint[1]][0] + 1])

    def get_chan_plot(self, chan_name, lap_stint=None):
        chan = self.channs[chan_name]
        meta = self.meta_map[chan.freq]
        data = chan.data
        if lap_stint == None or lap_stint[0] == None:
            return {"data": data, "dist": meta['dist'], "time": meta['time']}

        start = meta['lap_limits'][lap_stint[0]][0]
        end = meta['lap_limits'][lap_stint[0] + lap_stint[1]][0] + 1
        data = chan.data[start:end]
        dist = meta['dist'][start:end]

        time = meta['time'][start:end]
        return {"data": data, "dist": dist, "time":time}

    def get_lap_dist_time(self, lap_stint=None):
        meta = self.meta_map[max(self.meta_map.keys())]
        if lap_stint == None or lap_stint[0] == None:
            return {"dist": meta['dist'], "time": meta['time']}

        start = meta['lap_limits'][lap_stint[0]][0]
        end = meta['lap_limits'][lap_stint[0] + lap_stint[1]][0] + 1
        dist = meta['dist'][start:end]
        time = meta['time'][start:end]
        return {"dist": dist, "time":time}


    def get_damper_histogram(self, lap_stint=None, res=20, low=-510, high=510):
        lf = np.gradient(self.chan_data("sus_travel_lf", lap_stint=lap_stint), 0.005)
        rf = np.gradient(self.chan_data("sus_travel_rf", lap_stint=lap_stint), 0.005)
        lr = np.gradient(self.chan_data("sus_travel_lr", lap_stint=lap_stint), 0.005)
        rr = np.gradient(self.chan_data("sus_travel_rr", lap_stint=lap_stint), 0.005)

        global_min = None
        global_max = None
        max_y = None
        return_dict = {}
        for tp, name in zip((lf, rf, lr, rr), ("lf", "rf", "lr", "rr")):
            tp_min = np.amin(tp).round(int(-np.log10(res)) + 1)
            if global_min == None or tp_min < global_min:
                global_min = tp_min
            tp_max = np.amax(tp)
            bins = np.arange(low, high+1, res).round(1)
            if global_max == None or bins[-1] > global_max:
                global_max = bins[-1]
            freq, bins = np.histogram(tp, bins)
            count = tp.size
            max_freq = np.amax(freq) / count

            if max_y == None or max_freq > max_y:
                max_y = max_freq
            return_dict[name] = {"data": freq/count, "bin_edges": bins, "bin_centers": bins[:-1] + res / 2, "bins": bins.size - 1}

        return_dict["min_scale_x"] = low
        return_dict["max_scale_x"] = high
        return_dict["max_scale_y"] = np.round(max_y * 20 + 1) * 0.05
        return_dict["resolution"] = res

        return return_dict

    def analyse_damper_histogram(self, lap_stint, res=1):
        hist = self.get_damper_histogram(lap_stint, res)

        lf = hist["lf"]
        rf = hist["rf"]
        lr = hist["lr"]
        rr = hist["rr"]

        lf_rebound, lf_bump = np.split(lf["data"], [lf["bins"] // 2])
        lf_rebound = np.flip(lf_rebound)
        lf_bump_slow, lf_bump_fast = np.split(lf_bump, [40])
        lf_rebound_slow, lf_rebound_fast = np.split(lf_rebound, [40])

        lf_bump_slow_sum = np.sum(lf_bump_slow)
        lf_bump_fast_sum = np.sum(lf_bump_fast)
        lf_rebound_slow_sum = np.sum(lf_rebound_slow)
        lf_rebound_fast_sum = np.sum(lf_rebound_fast)

        rf_rebound, rf_bump = np.split(rf["data"], [rf["bins"] // 2])
        rf_rebound = np.flip(rf_rebound)
        rf_bump_slow, rf_bump_fast = np.split(rf_bump, [40])
        rf_rebound_slow, rf_rebound_fast = np.split(rf_rebound, [40])

        rf_bump_slow_sum = np.sum(rf_bump_slow)
        rf_bump_fast_sum = np.sum(rf_bump_fast)
        rf_rebound_slow_sum = np.sum(rf_rebound_slow)
        rf_rebound_fast_sum = np.sum(rf_rebound_fast)

        lr_rebound, lr_bump = np.split(lr["data"], [lr["bins"] // 2])
        lr_rebound = np.flip(lr_rebound)
        lr_bump_slow, lr_bump_fast = np.split(lr_bump, [40])
        lr_rebound_slow, lr_rebound_fast = np.split(lr_rebound, [40])

        lr_bump_slow_sum = np.sum(lr_bump_slow)
        lr_bump_fast_sum = np.sum(lr_bump_fast)
        lr_rebound_slow_sum = np.sum(lr_rebound_slow)
        lr_rebound_fast_sum = np.sum(lr_rebound_fast)

        rr_rebound, rr_bump = np.split(rr["data"], [rr["bins"] // 2])
        rr_rebound = np.flip(rr_rebound)
        rr_bump_slow, rr_bump_fast = np.split(rr_bump, [40])
        rr_rebound_slow, rr_rebound_fast = np.split(rr_rebound, [40])

        rr_bump_slow_sum = np.sum(rr_bump_slow)
        rr_bump_fast_sum = np.sum(rr_bump_fast)
        rr_rebound_slow_sum = np.sum(rr_rebound_slow)
        rr_rebound_fast_sum = np.sum(rr_rebound_fast)
        return {
            "lf": [lf_bump_slow, lf_bump_fast, lf_rebound_slow, lf_rebound_fast],
            "rf": [rf_bump_slow, rf_bump_fast, rf_rebound_slow, rf_rebound_fast],
            "lr": [lr_bump_slow, lr_bump_fast, lr_rebound_slow, lr_rebound_fast],
            "rr": [rr_bump_slow, rr_bump_fast, rr_rebound_slow, rr_rebound_fast]
        }


    def get_tyre_press_histogram(self, lap_stint=None, res=0.1):
        tp_lf = self.chan_data("tyre_press_lf", lap_stint=lap_stint)
        tp_rf = self.chan_data("tyre_press_rf", lap_stint=lap_stint)
        tp_lr = self.chan_data("tyre_press_lr", lap_stint=lap_stint)
        tp_rr = self.chan_data("tyre_press_rr", lap_stint=lap_stint)

        global_min = None
        global_max = None
        max_y = None
        return_dict = {}
        for tp, name in zip((tp_lf, tp_rf, tp_lr, tp_rr), ("lf", "rf", "lr", "rr")):
            tp_min = np.amin(tp).round(int(-np.log10(res)) + 1)
            if global_min == None or tp_min < global_min:
                global_min = tp_min
            tp_max = np.amax(tp)
            bins = np.arange(tp_min, tp_max+res, res).round(1)
            if global_max == None or bins[-1] > global_max:
                global_max = bins[-1]
            freq, bins = np.histogram(tp, bins)
            count = tp.size
            max_freq = np.amax(freq) / count

            if max_y == None or max_freq > max_y:
                max_y = max_freq
            return_dict[name] = {"data": freq/count, "bin_edges": bins, "bin_centers": bins[:-1] + res / 2}

        return_dict["min_scale_x"] = global_min - 2 * res
        return_dict["max_scale_x"] = global_max + 3 * res
        return_dict["max_scale_y"] = np.round(max_y * 20 + 1) * 0.05
        return_dict["resolution"] = res

        return return_dict 

    def get_tyre_temp_histogram(self, lap_stint=None, res=1):
        tp_lf = self.chan_data("tyre_tair_lf", lap_stint=lap_stint)
        tp_rf = self.chan_data("tyre_tair_rf", lap_stint=lap_stint)
        tp_lr = self.chan_data("tyre_tair_lr", lap_stint=lap_stint)
        tp_rr = self.chan_data("tyre_tair_rr", lap_stint=lap_stint)

        global_min = None
        global_max = None
        max_y = None
        return_dict = {}
        for tp, name in zip((tp_lf, tp_rf, tp_lr, tp_rr), ("lf", "rf", "lr", "rr")):
            tp_min = np.amin(tp)
            if global_min == None or tp_min < global_min:
                global_min = tp_min
            tp_max = np.amax(tp)
            bins = np.arange(tp_min, tp_max+res, res).round(1)
            if global_max == None or bins[-1] > global_max:
                global_max = bins[-1]
            freq, bins = np.histogram(tp, bins)
            count = tp.size
            max_freq = np.amax(freq) / count

            if max_y == None or max_freq > max_y:
                max_y = max_freq
            return_dict[name] = {"data": freq/count, "bin_edges": bins, "bin_centers": bins[:-1] + res / 2}

        return_dict["min_scale_x"] = global_min - 2 * res
        return_dict["max_scale_x"] = global_max + 3 * res
        return_dict["max_scale_y"] = np.round(max_y * 20 + 1) * 0.05
        return_dict["resolution"] = res

        return return_dict 


    def analyse_oversteer_understeer(self, lap_stint, track):
        steerangle_dict = self.get_chan_plot("steerangle", lap_stint=lap_stint)
        steerangle_dist = np.copy(steerangle_dict["dist"])
        steerangle_dist -= steerangle_dist[0]
        steerangle_sections = track.split_in_sections(np.absolute(steerangle_dict["data"]), steerangle_dist)

        oversteer_dict = self.get_chan_plot("oversteer", lap_stint=lap_stint)
        oversteer_dist = np.copy(oversteer_dict["dist"])
        oversteer_dist -= oversteer_dist[0]
        oversteer = np.interp(steerangle_dist, oversteer_dist, oversteer_dict["data"])

        oversteer_sections = track.split_in_sections(oversteer, steerangle_dist)




        return_array = [] 
        for index, data in enumerate(oversteer_sections):
            if data[0] == SectionType.STRAIGHT:
                continue

            length = data[1].size
            entry, mid, exit = np.split(data[1], [int(length * 0.35), int(length * 0.65)])
            #entry_calc, mid_calc, exit_calc = np.split(steerangle_calc_sections[index][1], [int(length * 0.25), int(length * 0.75)])


            entry = np.mean(entry)
            mid = np.mean(mid)
            exit = np.mean(exit)
            return_array.append([entry, mid, exit])

        return return_array



    def analyse_g_lat(self, lap_stint, track):
        g_lat_dict = self.get_chan_plot("g_lat", lap_stint=lap_stint)
        g_lat = g_lat_dict["data"]
        g_lat_dist = np.copy(g_lat_dict["dist"])
        g_lat_dist -= g_lat_dist[0]
        g_lat_abs = np.absolute(g_lat)
        g_lat_mean = np.mean(g_lat_abs)

        g_lat_sections = track.split_in_sections(g_lat_abs, g_lat_dist)
        return g_lat_sections


    def analyse_tc(self, lap_stint, track):
        tc_dict = self.get_chan_plot("tc", lap_stint=lap_stint)
        tc = tc_dict["data"]
        tc_dist = np.copy(tc_dict["dist"])
        tc_dist -= tc_dist[0]

        tc_sections = track.split_in_sections(tc, tc_dist)
        return tc_sections

    def analyse_tyre_press(self, lap_stint, track):
        tc_dict = self.get_chan_plot("tc", lap_stint=lap_stint)
        tc = tc_dict["data"]
        tc_dist = np.copy(tc_dict["dist"])
        tc_dist -= tc_dist[0]

        tc_sections = track.split_in_sections(tc, tc_dist)


        tp_lf_dict = self.get_chan_plot("tyre_press_lf", lap_stint=lap_stint)

        tp_lf = tp_lf_dict["data"]
        tp_rf = self.chan_data("tyre_press_rf", lap_stint=lap_stint)
        tp_lr = self.chan_data("tyre_press_lr", lap_stint=lap_stint)
        tp_rr = self.chan_data("tyre_press_rr", lap_stint=lap_stint)

        tp_dist = np.copy(tp_lf_dict["dist"])
        tp_dist -= tp_dist[0]

        tp_lf_sections = track.split_in_sections(tp_lf, tp_dist)
        tp_rf_sections = track.split_in_sections(tp_rf, tp_dist)
        tp_lr_sections = track.split_in_sections(tp_lr, tp_dist)
        tp_rr_sections = track.split_in_sections(tp_rr, tp_dist)

        tp_lf_sections = track.split_in_sections(tp_lf, tp_dist)
        tp_rf_sections = track.split_in_sections(tp_rf, tp_dist)
        tp_lr_sections = track.split_in_sections(tp_lr, tp_dist)
        tp_rr_sections = track.split_in_sections(tp_rr, tp_dist)

        tp_lf_corner = np.concatenate([item[1] if item[0] != SectionType.STRAIGHT else [] for item in tp_lf_sections], axis=0)
        tp_lf_corner_mean = np.mean(tp_lf_corner)
        tp_rf_corner = np.concatenate([item[1] if item[0] != SectionType.STRAIGHT else [] for item in tp_rf_sections], axis=0)
        tp_rf_corner_mean = np.mean(tp_rf_corner)
        tp_lr_corner = np.concatenate([item[1] if item[0] != SectionType.STRAIGHT else [] for item in tp_lr_sections], axis=0)
        tp_lr_corner_mean = np.mean(tp_lr_corner)
        tp_rr_corner = np.concatenate([item[1] if item[0] != SectionType.STRAIGHT else [] for item in tp_rr_sections], axis=0)
        tp_rr_corner_mean = np.mean(tp_rr_corner)


        tp_lf_straight = np.concatenate([item[1] if item[0] == SectionType.STRAIGHT else [] for item in tp_lf_sections], axis=0)
        tp_lf_straight_mean = np.mean(tp_lf_straight)
        tp_rf_straight = np.concatenate([item[1] if item[0] == SectionType.STRAIGHT else [] for item in tp_rf_sections], axis=0)
        tp_rf_straight_mean = np.mean(tp_rf_straight)
        tp_lr_straight = np.concatenate([item[1] if item[0] == SectionType.STRAIGHT else [] for item in tp_lr_sections], axis=0)
        tp_lr_straight_mean = np.mean(tp_lr_straight)
        tp_rr_straight = np.concatenate([item[1] if item[0] == SectionType.STRAIGHT else [] for item in tp_rr_sections], axis=0)
        tp_rr_straight_mean = np.mean(tp_rr_straight)
        

        tp_lf_total = np.concatenate([item[1] for item in tp_lf_sections], axis=0)
        tp_lf_total_mean = np.mean(tp_lf_total)
        tp_rf_total = np.concatenate([item[1] for item in tp_rf_sections], axis=0)
        tp_rf_total_mean = np.mean(tp_rf_total)
        tp_lr_total = np.concatenate([item[1] for item in tp_lr_sections], axis=0)
        tp_lr_total_mean = np.mean(tp_lr_total)
        tp_rr_total = np.concatenate([item[1] for item in tp_rr_sections], axis=0)
        tp_rr_total_mean = np.mean(tp_rr_total)

        return {
                "lf" : [tp_lf_corner, tp_lf_corner_mean, tp_lf_total, tp_lf_total_mean],
                "rf" : [tp_rf_corner, tp_rf_corner_mean, tp_rf_total, tp_rf_total_mean],
                "lr" : [tp_lr_corner, tp_lr_corner_mean, tp_lr_total, tp_lr_total_mean],
                "rr" : [tp_rr_corner, tp_rr_corner_mean, tp_rr_total, tp_rr_total_mean]
            }



    def get_stint_profile(self, performance_array, lap_stint, track):
        dist_time_map = self.get_lap_dist_time()
        dist = dist_time_map["dist"]
        time = dist_time_map["time"]
        lap_profile = track.build_lap_profile(dist, time)
        corner_count = 0
        section_profile = {
            SectionType.CORNER_SLOW: [0, 0, 0],
            SectionType.CORNER_MED: [0, 0, 0],
            SectionType.CORNER_FAST: [0, 0, 0]
        }
        for section, time in lap_profile["section_times"]:
            if section != SectionType.STRAIGHT:
                section_profile[section][0] += (time * performance_array[corner_count][0])
                section_profile[section][1] += (time * performance_array[corner_count][1])
                section_profile[section][2] += (time * performance_array[corner_count][2])
                corner_count+=1

        for section, results in section_profile.items():
            total_time = lap_profile["total_section_times"][section]
            section_profile[section] = [r / total_time for r in results]

        array = section_profile[SectionType.CORNER_SLOW] + section_profile[SectionType.CORNER_MED] + section_profile[SectionType.CORNER_FAST]
        return np.array(array)


from enum import Enum
import bisect
class SectionType(Enum):
    STRAIGHT = 1
    CORNER_SLOW = 2
    CORNER_MED = 3
    CORNER_FAST = 4

    def description(self):
        if self == SectionType.STRAIGHT:
            return "Straight"
        elif self == SectionType.CORNER_SLOW:
            return "Slow Speed Corner"
        elif self == SectionType.CORNER_MED:
            return "Medium Speed Corner"
        elif self == SectionType.CORNER_FAST:
            return "High Speed Corner"




class Track():
    def __init__(self, name, length, section_ends = [], section_names = [], section_types=[]):
        self.name = name
        self.length = length
        self.section_ends = section_ends
        self.section_names = section_names
        self.section_types = section_types
        #self.section_types = [SectionType.CORNER if name[:4] == "Turn" else SectionType.STRAIGHT for name in section_names]

    @classmethod
    def from_sections(name, section_ends, section_names):
        if len(section_names) != len(section_ends): return None
        return cls(name, section_ends, section_names)

    def add_section(self, section_end, section_name, section_type):
        if section_end in self.section_ends:
            return False
        idx = bisect.bisect_left(self.section_ends, section_end)
        self.section_ends.insert(idx, section_end)
        self.section_names.insert(idx, section_name)

        self.section_names.insert(idx, SectionType.CORNER if section_name[:4] == "Turn" else SectionType.STRAIGHT)

    def get_sections(self):
        if len(self.section_names) == 0:
            return (0, "Track")

        return zip(self.section_ends, self.section_names)


    def __is_corner(self, dist):
        if dist > self.length or dist < 0:
            return None
        for index, checkpoint in enumerate(self.section_ends):
            if dist < checkpoint:
                return section_types[index] == SectionType.CORNER
        return section_types[0] == SectionType.CORNER

    def get_corner_dist_map(self):
        pass


    def build_lap_profile(self, dist_array, time_array):
        breakpoints = np.rint(np.interp(self.section_ends, dist_array, np.arange(dist_array.size))).astype(int)
        times = time_array[breakpoints]
        times = np.concatenate((np.array([time_array[0]]),times))


        total_section_times = {}
        section_times = []
        for i, section in enumerate(self.section_types):
            delta = times[i+1] - times[i]
            section_times.append((section, delta))
            total_section_times[section] = total_section_times.get(section, 0) + delta
            
        delta =  + (time_array[-1] - time_array[0] - times[-1])
        time_section_0 = section_times[0][1] + delta
        section_times[0] = (section_times[0][0], time_section_0)
        total_section_times[self.section_types[0]] = total_section_times.get(self.section_types[0], 0) + delta

        return {"section_times": section_times, "total_section_times": total_section_times}

    def split_in_sections(self, data_array, dist_array):
        # working
        # test split of channels into laps
        breakpoints = np.rint(np.interp(self.section_ends, dist_array, np.arange(dist_array.size))).astype(int)
        #np.where(breakpoints >= dist_array.size, breakpoints, dist_array.size-1)
        sections = np.split(data_array, breakpoints)

        sections_array = list(zip(self.section_types, sections))
        sections_array.append((self.section_types[0], sections[-1]))
        return sections_array



