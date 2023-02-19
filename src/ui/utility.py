import dearpygui.dearpygui as dpg
import numpy as np

class UiHistogram:
        
    def get_tags(self):
        name = self.name
        return (
            [f"{name}_plot_lf", f"{name}_plot_rf", f"{name}_plot_lr", f"{name}_plot_rr"],
            [f"{name}_x_lf", f"{name}_x_rf", f"{name}_x_lr", f"{name}_x_rr"],
            [f"{name}_y_lf", f"{name}_y_rf", f"{name}_y_lr", f"{name}_y_rr"],
            [f"{name}_data_lf", f"{name}_data_rf", f"{name}_data_lr", f"{name}_data_rr"]
        )

    def get_tags_x(self):
        name = self.name
        return (
           f"{name}_x_lf", f"{name}_x_rf", f"{name}_x_lr", f"{name}_x_rr"
        )

    def get_tags_y(self):
        name = self.name
        return (
           f"{name}_y_lf", f"{name}_y_rf", f"{name}_y_lr", f"{name}_y_rr"
        )

    def get_tags_plot(self):
        name = self.name
        return (
           f"{name}_plot_lf", f"{name}_plot_rf", f"{name}_plot_lr", f"{name}_plot_rr"
        )

    def get_tags_data(self):
        name = self.name
        return (
           f"{name}_data_lf", f"{name}_data_rf", f"{name}_data_lr", f"{name}_data_rr"
        )


    def __init__(self, name, x_axis_label, y_axis_label):
        self.name = name 

        labels = ["Left Front", "Right Front", "Left Rear", "Right Rear"]
        plot, x, y, data = self.get_tags()
           
        #creation
        with dpg.stage() as self._staging_container_id:
            with dpg.group():
                with dpg.group(horizontal=True):
                    with dpg.plot(label="Left Front", height=350, width=0.5, tag=plot[0]):
                        dpg.add_plot_legend()

                        # create x axis
                        dpg.add_plot_axis(dpg.mvXAxis, label=x_axis_label, no_gridlines=True, tag=x[0])
                    
                        # create y axis
                        with dpg.plot_axis(dpg.mvYAxis, label=y_axis_label, tag=y[0]):
                            dpg.add_bar_series((), (), label=x_axis_label, weight=0.2, tag=data[0])
                    with dpg.plot(label="Right Front", height=350, width=0.5, tag=plot[1]):
                        dpg.add_plot_legend()

                        # create x axis
                        dpg.add_plot_axis(dpg.mvXAxis, label=x_axis_label, no_gridlines=True, tag=x[1])
            
                        # create y axis
                        with dpg.plot_axis(dpg.mvYAxis, label=y_axis_label, tag=y[1]):
                            dpg.add_bar_series((),(), label=x_axis_label, weight=0.2, tag=data[1])
                with dpg.group(horizontal=True):
                    with dpg.plot(label="Left Rear", height=350, width=0.5, tag=plot[2]):

                        dpg.add_plot_legend()

                        # create x axis
                        dpg.add_plot_axis(dpg.mvXAxis, label=x_axis_label, no_gridlines=True, tag=x[2])
            
                        # create y axis
                        with dpg.plot_axis(dpg.mvYAxis, label=y_axis_label, tag=y[2]):
                            dpg.add_bar_series((), (), label=x_axis_label, weight=0.2, tag=data[2])

                    with dpg.plot(label="Right Rear", height=350, width=0.5, tag=plot[3]):

                        dpg.add_plot_legend()

                        # create x axis
                        dpg.add_plot_axis(dpg.mvXAxis, label=x_axis_label, no_gridlines=True, tag=x[3])
                        #dpg.set_axis_ticks(dpg.last_item(), (("S1", 11), ("S2", 21), ("S3", 31)))
            
                        # create y axis
                        with dpg.plot_axis(dpg.mvYAxis, label=y_axis_label, tag=y[3]):
                            dpg.add_bar_series((), (), label=x_axis_label, weight=0.2, tag=data[3])
                            #dpg.add_bar_series([11, 21, 31], [83, 75, 72], label="Midterm Exam", weight=1)
                            #dpg.add_bar_series([12, 22, 32], [42, 68, 23], label="Course Grade", weight=1)

    @classmethod
    def create(cls, name, x_axis_label=None, y_axis_label=None):
        if x_axis_label == None: x_axis_label = name + "x_axis"
        if y_axis_label == None: y_axis_label = name + "y_axis"
        return cls(name, x_axis_label, y_axis_label)

    @classmethod
    def get(cls, name):
        return cls.histograms[name]

    def unstage(self, parent):
        dpg.push_container_stack(parent)
        dpg.unstage(self._staging_container_id)
        dpg.pop_container_stack()

    def unstage(self):
        dpg.unstage(self._staging_container_id)

    def set_resolution(self, resolution):
        for data in self.get_tags_data():
            dpg.configure_item(data, weight=resolution)


    def update_values(self, data):
        for index, value in enumerate(self.get_tags_data()):
            dpg.set_value(value, data[index])

    def update_y_axis(self, lower, upper):
        for y in self.get_tags_y():
            dpg.set_axis_limits(y, lower, upper)

    def update_x_axis(self, lower, upper):
        for y in self.get_tags_x():
            dpg.set_axis_limits(y, lower, upper)

    def update_histogram(self, lower_x, upper_x, lower_y, upper_y, data):
        self.update_x_axis(lower_x, upper_x)
        self.update_y_axis(lower_y, upper_y)
        self.update_values(data)


    @classmethod
    def update_histogram_by_name(cls, lower_x, upper_x, lower_y, upper_y, data):
        pass


class UiGraph():
    def __init__(self, name, x_axis_label, y_axis_label):
        self.name = name

        self.tags = {
            name+"_plot": "",
            name+"_data": "",
            name+"_x": "",
            name+"_y": "",
        }
        self.plot_tag = name+"_plot"
        self.data_tag = name+"_data"
        self.x_tag = name+"_x"
        self.y_tag = name+"_y"

        #creation
        with dpg.stage() as self._staging_container_id: 
            with dpg.plot(label="Right Rear", height=230, width=1800) as plot_tag:
                self.tags[self.plot_tag] = plot_tag
                dpg.add_plot_legend()

                # create x axis
                self.tags[self.x_tag] = dpg.add_plot_axis(dpg.mvXAxis, label="Distance, [m]", no_gridlines=True)
                #dpg.set_axis_ticks(dpg.last_item(), (("S1", 11), ("S2", 21), ("S3", 31)))
    
                # create y axis
                with dpg.plot_axis(dpg.mvYAxis, label=y_axis_label, tag=self.y_tag) as y_tag:
                    self.tags[self.y_tag] = y_tag
                    #dpg.add_line_series(sindatax, sindatay, label="0.5 + 0.5 * sin(x)")
                    self.tags[self.data_tag] = dpg.add_line_series((), (), label=x_axis_label)

        self.plot_tag = self.tags[self.plot_tag]
        self.data_tag = self.tags[self.data_tag]
        self.x_tag = self.tags[self.x_tag]
        self.y_tag = self.tags[self.y_tag]

    @classmethod
    def create(cls, name, x_axis_label=None, y_axis_label=None):
        if x_axis_label == None: x_axis_label = name + "x_axis"
        if y_axis_label == None: y_axis_label = name + "y_axis"
        return cls(name, x_axis_label, y_axis_label)

    @classmethod
    def get(cls, name):
        return cls.histograms[name]

    def unstage(self, parent=None):
        if parent != None:
            dpg.push_container_stack(parent)
            dpg.unstage(self._staging_container_id)
            dpg.pop_container_stack()
        else:
            dpg.unstage(self._staging_container_id)

    def update_values(self, data):
        dpg.set_value(self.data_tag, data)

    def update_y_axis(self, lower, upper):
        dpg.set_axis_limits(self.y_tag, lower, upper)

    def update_x_axis(self, lower, upper):
        dpg.set_axis_limits(self.x_tag, lower, upper)

    def update(self, lower_x, upper_x, lower_y, upper_y, data):
        self.update_x_axis(lower_x, upper_x)
        self.update_y_axis(lower_y, upper_y)
        self.update_values(data)

class UiSubplot():
    def __init__(self, name, y_axis_label, x_axis_label):
        self.name = name
        self.plot_tag = name+"_plot"
        self.x_tag = name+"_x"
        self.y_tag = name+"_y"
        self.data_channs = {}

        with dpg.stage() as self._staging_container_id: 
            with dpg.plot(height=230, width=1800, tag=self.plot_tag):
                dpg.add_plot_legend()

                # create x axis
                dpg.add_plot_axis(dpg.mvXAxis, label=x_axis_label, tag=self.x_tag)
                #dpg.set_axis_ticks(dpg.last_item(), (("S1", 11), ("S2", 21), ("S3", 31)))
    
                # create y axis
                with dpg.plot_axis(dpg.mvYAxis, label=y_axis_label, tag=self.y_tag):
                    #dpg.add_line_series(sindatax, sindatay, label="0.5 + 0.5 * sin(x)")
                    #dpg.add_line_series((), (), label=x_axis_label, tag=self.data_tag)
                    pass

    def unstage(self, parent=None):
        if parent != None:
            dpg.push_container_stack(parent)
            dpg.unstage(self._staging_container_id)
            dpg.pop_container_stack()
        else:
            dpg.unstage(self._staging_container_id)

    def add_chan(self, chan_name, line_label):
        if chan_name in self.data_channs:
            return

        dpg.push_container_stack(self.y_tag)
        tag = dpg.add_line_series((), (), label=line_label)
        dpg.configure_item(tag, show=False)
        self.data_channs[chan_name] = tag
        dpg.pop_container_stack()

    def del_chan(self, chan_name):
        if chan_name in self.data_channs:
            dpg.delete_item(self.data_channs[chan_name])
            del self.data_channs[chan_name]

    def update_x_axis(self, lower, upper):
        dpg.set_axis_limits(self.x_tag, lower, upper)

    def update_y_axis(self, lower, upper):
        dpg.set_axis_limits(self.y_tag, lower, upper)

    def update_chan_value(self, chan_name, data_x, data_y):
        if not chan_name in self.data_channs:
            return
        self.del_chan(chan_name)
        self.add_chan(chan_name, chan_name)

        line = self.data_channs[chan_name]
        dpg.set_value(line, (data_x, data_y))
        if data_x == () or data_y == ():
            dpg.configure_item(line, show=False)
        else:
            dpg.configure_item(line, show=True)


class UiPlot():
    def __init__(self, name, length, label, x_axis_label):
        self.name = name
        self.plot_tag = name+"_plot"
        self.x_tag = name+"_x"
        self.y_tag = name+"_y"
        self.subplots = {}
        self.x_axis_label = x_axis_label

        #creation
        with dpg.stage() as self._staging_container_id: 
            with dpg.subplots(length, 1, label=label, height=1500, width=-1, tag=self.plot_tag, link_all_x=True):
                pass


    @classmethod
    def create(cls, name, x_axis_label=None, y_axis_label=None):
        if x_axis_label == None: x_axis_label = name + "x_axis"
        if y_axis_label == None: y_axis_label = name + "y_axis"
        return cls(name, x_axis_label, y_axis_label)

    @classmethod
    def get(cls, name):
        return cls.histograms[name]

    def unstage(self, parent=None):
        if parent != None:
            dpg.push_container_stack(parent)
            dpg.unstage(self._staging_container_id)
            dpg.pop_container_stack()
        else:
            dpg.unstage(self._staging_container_id)

    def add_subplot(self, unit, y_axis_label):
        if unit in self.subplots:
            return

        dpg.push_container_stack(self.plot_tag)
        subplot = UiSubplot(unit, y_axis_label, self.x_axis_label)
        subplot.unstage()
        dpg.pop_container_stack()

        self.subplots[unit] = subplot

    def del_subplot(self, unit):
        if unit in self.data_channs:
            dpg.delete_item(self.data_channs[unit])
            del self.data_channs[unit]

    def update_values(self, data):
        dpg.set_value(self.data_tag, data)

    def update_y_axis(self, lower, upper):
        dpg.set_axis_limits(self.y_tag, lower, upper)

    def update_x_axis(self, lower, upper):
        dpg.set_axis_limits(self.x_tag, lower, upper)

    def update(self, lower_x, upper_x, lower_y, upper_y, data):
        self.update_x_axis(lower_x, upper_x)
        self.update_y_axis(lower_y, upper_y)
        self.update_values(data)

    def add_chan(self, unit, chan_name, line_label):
        if unit not in self.subplots:
            return

        self.subplots[unit].add_chan(chan_name, line_label)
    def del_chan(self, chan_name):
        if unit not in self.subplots:
            return

        self.subplots[unit].del_chan(unit, chan_name)

    def update_chan_value(self, chan_name, unit, data_x, data_y):
        if unit not in self.subplots:
            return

        if data_x != () and data_y != ():
            data_x = np.copy(data_x)
            data_x -= data_x[0]

        self.subplots[unit].update_chan_value(chan_name, data_x, data_y)

        if data_x != () and data_y != ():
            y_max = data_y.max()
            y_min = data_y.min()   

            diff = y_max-y_min
            #self.subplots[unit].update_y_axis(y_min - 0.1*diff, y_max + 0.1 * diff)
            self.subplots[unit].update_x_axis(data_x[0], data_x[-1])


