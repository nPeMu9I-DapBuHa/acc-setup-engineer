import dearpygui.dearpygui as dpg
import numpy as np
import json

import src.ldparser.ldparser as ldparser
import src.ldparser.q_learning as q_learning
import src.setup as setuploader
import src.ui.utility as util
import src.inference as inf

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]



class CheckboxGroup():
    groups={}

    def __init__(self, name):
        self.group = []
        self.name = name


    @classmethod
    def create(cls, name):
        group = cls(name)
        cls.groups[name] = group
        return group

    @classmethod
    def get(cls, name):
        return cls.groups[name]

    def add_checkbox(self, checkbox):
        self.group.append(checkbox)

    def select(self, checkbox):
        for option in self.group:
            dpg.set_value(option, False)
        dpg.set_value(checkbox, True)




class Variables():
    vars = {}

    @classmethod
    def get_var(cls, var_name):
        try:
            return cls.vars[var_name]
        except KeyError:
            return None

    @classmethod
    def set_var(cls, var_name, var):
        cls.vars[var_name] = var

    @classmethod
    def del_val(cls, val_name):
        del cls.val[val_name]

    @classmethod
    def initialise_vars(cls):
        with open("./config/config.json") as f:
            cls.set_var("config", json.load(f))


        cls.set_var("stint_mode", False)
        cls.set_var("telemetry", None)
        cls.set_var("graphs", {})
        cls.set_var("last_state", None)
        cls.set_var("nn_choose_action_btns", [])

        spa = ldparser.Track("Spa", 6957, 
            [
                367, 
                418,  
                1095, 
                1210,
                1305, 
                1340,
                2379, 
                2450,
                2477, 
                2545,
                2596, 
                2722,
                2974, 
                3125,
                3241, 
                3345,
                3749, 
                4130,
                4436, 
                4550,
                4597, 
                4707,
                4886,
                4969,
                5074,
                5243,
                6089, 
                6269,
                6704,
                6753,
                6798
            ],
            [
                "Str 0-1",
                "Turn 1",
                "Str 1-2",
                "Turn 2",
                "Str 2-3",
                "Turn 3",
                "Str 3-4",
                "Turn 4",
                "Str 4-5",
                "Turn 5",
                "Str 5-6",
                "Turn 6",
                "Str 6-7",
                "Turn 7",
                "Str 7-8",
                "Turn 8",
                "Str 8-9",
                "Turn 9",
                "Str 9-10",
                "Turn 10",
                "Str 10-11",
                "Turn 11",
                "Str 11-12",
                "Turn 12",
                "Str 12-13",
                "Turn 13",
                "Str 13-14",
                "Turn 14",
                "Str 14-15",
                "Turn 15",
                "Turn 16",
            ],
            [
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_SLOW,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_FAST,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_FAST,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_MED,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_MED,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_MED,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_SLOW,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_MED,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_FAST,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_MED,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_MED,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_MED,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_FAST,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_FAST,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_SLOW,
                ldparser.SectionType.CORNER_SLOW
            ])

        imola = ldparser.Track("Imola", 4869, 
            [
                662, 
                765,  
                828, 
                861,
                947, 
                1292,
                1363, 
                1388,
                1474, 
                1672,
                1772, 
                2273,
                2461, 
                2697,
                2887, 
                3319,
                3366, 
                3401,
                3906, 
                3964,
                4084,
                4162, 
                4210,
                4307
            ],
            [
                "Str 0-1",
                "Turn 1",
                "Turn 2",
                "Str 2-3",
                "Turn 3",
                "Str 3-4",
                "Turn 4",
                "Str 4-5",
                "Turn 5",
                "Str 5-6",
                "Turn 6",
                "Str 6-7",
                "Turn 7",
                "Str 7-8",
                "Turn 8",
                "Str 8-9",
                "Turn 9",
                "Turn 10",
                "Str 10-11",
                "Turn 11",
                "Str 11-12",
                "Turn 12",
                "Str 12-13",
                "Turn 13"
            ],
            [
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_MED,
                ldparser.SectionType.CORNER_MED,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_FAST,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_FAST,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_MED,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_SLOW,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_FAST,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_MED,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_SLOW,
                ldparser.SectionType.CORNER_SLOW,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_FAST,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_SLOW,
                ldparser.SectionType.STRAIGHT,
                ldparser.SectionType.CORNER_SLOW,
            ])
        cls.set_var("track", imola)

        if Variables.get_var("config")["track"] == "spa":
            cls.set_var("track", spa)
        elif Variables.get_var("config")["track"] == "imola":
            cls.set_var("track", imola)
        else:
            # default: imola
            cls.set_var("track", imola)

        agent = q_learning.Agent(gamma=0.99, epsilon = 1.0, batch_size = 10, n_outputs = 34, eps_end = 0.01,
            input_dims=[9], lr=0.0001)
        agent.load()

        cls.set_var("agent", agent)

        cls.set_var("inference", inf.InferenceSystem())
        cls.set_var("tyre_optimiser", inf.TyreOptimiser())
        cls.set_var("damper_optimiser", inf.DamperOptimiser())

class FunctionManager():

    @staticmethod
    def load_ld_file(ld_path):
        head_, chans = ldparser.read_ldfile(ld_path)
        laps = np.array(ldparser.laps(ld_path))
        ds = ldparser.MyLDDataStore(
            chans, laps, Variables.get_var("config")["steer_ratio"]
        )
        return ds

    @staticmethod
    def load_setup(file_path):
        return setuploader.Setup.from_file(file_path)

    


class CallbackManager(metaclass=Singleton):
    callbacks = {}

    @classmethod
    def get_callback(cls, callback_name):
        return cls.callbacks[callback_name]

    @classmethod
    def add_callback(cls, callback_name, callback):
        cls.callbacks[callback_name] = callback

    @classmethod
    def del_callback(cls, callback_name):
        del cls.callbacks[callback_name]

    @classmethod
    def initialise_callbacks(cls):

        def resize_window(sender, app_data, user_data=None): ########### scaling -----------------------------------------------------------------------------------------------------------------------
            print(dpg.get_item_height(app_data))
            print(dpg.get_item_width(app_data))
            #  dpg.set_item_indent(b1, 100), width, height

        cls.add_callback("resize_window", resize_window)

        def load_setup_file(sender, app_data, user_data=None):
            file_path = app_data["file_path_name"]
            data = FunctionManager.load_setup(file_path)
            Variables.set_var("setup", {"setup" : data, "name" : app_data["file_name"][:-5]})
            WindowManager.telemetry_window_load_setup()
        cls.add_callback("load_setup_file", load_setup_file)

        def load_file(sender, app_data, user_data=None):
            print("Sender: ", sender)
            print("App Data: ", app_data)
            file_path = app_data["file_path_name"]
            if file_path[-3:] != ".ld":
                if file_path[-2:] == ".*": file_path = file_path[:-2] + ".ld"
                else: raise RuntimeError("Incorrect file selected") 
            a = FunctionManager.load_ld_file(file_path)
            file_name = app_data["file_name"]
            file_name = file_name if file_name[-3:] == ".ld" else file_name[:-2] + ".ld"
            Variables.set_var("telemetry", {"data" : a, "name" : file_name})
            WindowManager.telemetry_window_load_telemetry()
        cls.add_callback("load_file", load_file)

        def swap_window(sender, app_data, user_data=None):
            if user_data == None:
                print("Error swapping windows")
                return
            WindowManager.set_main_window(user_data)
        cls.add_callback("swap_window", swap_window)

        def select_stint_mode(sender):
            pass
        cls.add_callback("select_stint_mode", select_stint_mode)

        def select_checkbox(sender, app_data=None, user_data=None):
            if user_data["group"] == "lap_select":
                group = CheckboxGroup.get("lap_select")
                group.select(sender)
                Variables.set_var("selected_lap", user_data["value"])
                WindowManager.update_telemerty_graphs()
                WindowManager.display_lap_setion_table()
        cls.add_callback("select_checkbox", select_checkbox)

        def select_motec_channel(sender, app_data=None, user_data=None):
            value = dpg.get_value(sender)
            Variables.get_var("motec_channels")[user_data["chan_name"]] = value
            telemetry = Variables.get_var("telemetry")["data"]
            data = telemetry.get_chan_plot(user_data["chan_name"], lap_stint=(Variables.get_var("selected_lap"), 1))
            unit = telemetry.get_chan_unit(user_data["chan_name"])
            plot = Variables.get_var("plot")
            if not value:
                plot.update_chan_value(user_data["chan_name"], unit, (), ())
            else:
                plot.update_chan_value(user_data["chan_name"], unit, data["dist"], data["data"])
        cls.add_callback("select_motec_channel", select_motec_channel)

        def analyse_telemetry(sender, app_data=None, user_data=None):
            return WindowManager.analyse_telemetry()
        cls.add_callback("analyse_telemetry", analyse_telemetry)

        def corner_analysis(sender, app_data=None, user_data=None):
            corner_num = Variables.get_var("corner_count")
            telemetry = Variables.get_var("telemetry")["data"]
            track = Variables.get_var("track")
            lap = Variables.get_var("selected_lap")
            if lap == None:
                return
            else:
                lap_stint = (lap, 1)
            lap_time = telemetry.laps_times[lap]

            corner_array = []
            for i in range(1, corner_num + 1):
                corner_array.append((dpg.get_value(f"corner_{i}_entry"), dpg.get_value(f"corner_{i}_mid"), dpg.get_value(f"corner_{i}_exit")))

            agent = Variables.get_var("agent")
            setup = Variables.get_var("setup")
            inference = Variables.get_var("inference")
            


            steer_profile = telemetry.get_stint_profile(corner_array, lap_stint, track)
            reward = agent.calc_reward(steer_profile, lap_time, 100.000)
            state, action = agent.choose_action(steer_profile, setup["setup"].data)

            if reward is None:
                agent.store_last_state(state, lap_time)
                return

            agent.store_transition(state, reward, lap_time)
            agent.learn()
            agent.save()
            
        cls.add_callback("corner_analysis", corner_analysis)

        def apply_changes_tyres(sender, app_data=None, user_data=None):
            tyre_optimiser = Variables.get_var("tyre_optimiser")
            slider_ids = user_data["slider_ids"]
            action = {
                "lf": dpg.get_value(slider_ids["lf"]),
                "rf": dpg.get_value(slider_ids["rf"]),
                "lr": dpg.get_value(slider_ids["lr"]),
                "rr": dpg.get_value(slider_ids["rr"]),
            }
            tyre_optimiser.store_action(user_data["pressures"], action)
            dpg.configure_item("store_results_tyres_button", enabled=False)

        cls.add_callback("apply_changes_tyres", apply_changes_tyres)

        def refresh_corner_performance_table_button(sender, app_data=None, user_data=None):
            WindowManager.display_lap_setion_table()
        cls.add_callback("refresh_corner_performance_table_button", refresh_corner_performance_table_button)

        def store_results_tyres(sender, app_data=None, user_data=None):
            tyre_optimiser = Variables.get_var("tyre_optimiser")
            tyre_optimiser.store_results(user_data["pressures"])

        cls.add_callback("store_results_tyres", store_results_tyres)

        def optimise_pressure(sender, app_data=None, user_data=None):
            """telemetry = Variables.get_var("telemetry")["data"]
            track = Variables.get_var("track")
            lap = Variables.get_var("selected_lap")
            if lap == None:
                return
            else:
                lap_stint = (lap, 1)
            lap_time = telemetry.laps_times[lap]

            print(telemetry.analyse_oversteer_understeer(lap_stint, track))

            agent = Variables.get_var("agent")
            tyre_optimiser = Variables.get_var("tyre_optimiser")
            pressures = telemetry.analyse_tyre_press(lap_stint, track)

            actions = tyre_optimiser.infer(pressures)
            print(actions)"""
            WindowManager.telemetry_window_update_tyre_report()
            
        cls.add_callback("optimise_pressure", optimise_pressure)

        def optimise_dampers(sender, app_data=None, user_data=None):
            WindowManager.telemetry_window_update_damper_report()
        cls.add_callback("optimise_dampers", optimise_dampers)


        def engineer_report(sender, app_data=None, user_data=None):
            WindowManager.engineer_report()
        cls.add_callback("engineer_report", engineer_report)

        def nn_choose_action(sender, app_data=None, user_data=None):
            print("Chose action")
            Variables.set_var("last_state", user_data)
            for btn in Variables.get_var("nn_choose_action_btns"):
                dpg.configure_item(btn, enabled=False, show=False)

            WindowManager.update_last_state_table()
        cls.add_callback("nn_choose_action", nn_choose_action)

        def nn_learn(sender, app_data=None, user_data=None):
            print("Learning")

            last_state = Variables.get_var("last_state")
            current_state = Variables.get_var("current_state")

            if last_state is None or current_state is None:
                print("Unable to learn")
                return

            for btn in Variables.get_var("nn_learn_btn"):
                dpg.configure_item(btn, enabled=False, show=False)

            agent = Variables.get_var("agent")
            reward = agent.calc_reward(last_state["profile"], last_state["lap_time"], current_state["profile"], current_state["lap_time"], 136.000)

            agent.store_transition(last_state["profile"], last_state["chosen_action"], reward)
            agent.learn()
            agent.save()
            WindowManager.update_last_state_table()
        cls.add_callback("nn_learn", nn_learn)

             
        

class WindowManager(metaclass=Singleton):
    windows = []

    @classmethod
    def get_window(cls, window_name):
        return cls.windows[window_name]

    @classmethod
    def add_window(cls, window_name):
        cls.windows.append(window_name)
        dpg.configure_item(window_name, show=False)

    @classmethod
    def del_Window(cls, window_name):
        cls.windows.remove(window_obj)

    @classmethod
    def set_main_window(cls, window_name):
        if not window_name in cls.windows:
            print("Error")
        else:
            for w in cls.windows:
                dpg.configure_item(w, show=False)
            dpg.configure_item(window_name, show=True)
            dpg.set_primary_window(window_name, True)

    @classmethod
    def initialise_windows(cls):
        with dpg.item_handler_registry(tag="widget handler") as handler:
            dpg.add_item_resize_handler(callback=CallbackManager.get_callback("resize_window"))


        with dpg.file_dialog(directory_selector=False, show=False, width=800, height=600, callback=CallbackManager.get_callback("load_file"), tag="telemetry_file_dialog"):
            dpg.add_file_extension(".ld", color=(0, 255, 0, 255))
            dpg.add_file_extension(".lnk", color=(0, 0, 255, 255))

        with dpg.file_dialog(directory_selector=False, show=False, width=800, height=600, callback=CallbackManager.get_callback("load_setup_file"), tag="setup_file_dialog"):
            dpg.add_file_extension(".json", color=(255, 0, 0, 255))
            dpg.add_file_extension(".lnk", color=(0, 0, 255, 255))

        with dpg.window(tag="main_window"):
            WindowManager.add_window("main_window")
            dpg.add_text("ACC race engineer")
            dpg.add_button(label="Start", callback=CallbackManager.get_callback("swap_window"), user_data="telemetry_window")
            #dpg.add_button(label="Load Telemetry file", callback=lambda: dpg.show_item("file_dialog_tag"))
            #dpg.add_button(label="Load Setup", callback=lambda: dpg.show_item("file_dialog_tag"))

        with dpg.window(tag="telemetry_window", width=1920, height=1080, show=False):
            WindowManager.add_window("telemetry_window")
            #with dpg.group(horizontal=True):
            #    with dpg.group(tag="Group1"):
            dpg.add_text("ACC race engineer Telemetry window")
            dpg.add_text("Telemetry file: None", tag="telemetry_window_telemetry_file_name_text")
            dpg.add_button(label="Load Telemetry file", callback=lambda: dpg.show_item("telemetry_file_dialog"))
            dpg.add_text("Setup file: None", tag="telemetry_window_setup_file_name_text")
            dpg.add_button(label="Load Setup file", callback=lambda: dpg.show_item("setup_file_dialog"))

            #with dpg.group(horizontal=True):
            #    dpg.add_text("Select stint length")
            #    #dpg.add_slider_int(label="laps", min_value=1, max_value=10, default_value=1, callback=lambda x: x)
            #    dpg.add_combo([i for i in range(1, 10)], default_value=1, label="laps", width=200)
            #rb = dpg.add_checkbox(callback=CallbackManager.get_callback("select_stint_mode"))
            #print(dpg.get_item_children("telemetry_window", 1))

            with dpg.tab_bar(label="Telemetry", tag="tb1"):
                with dpg.tab(label="Lap Table", tag="lap_table_tab"):
                    pass
                with dpg.tab(label="Tyres"):
                    with dpg.tab_bar():
                        with dpg.tab(label="Tyre pressures"):
                            histogram = util.UiHistogram.create("tyre_press_histogram", u"Tyre pressure, [psi]", "Proportion")
                            Variables.get_var("graphs")["tyre_press_histogram"] = histogram
                            histogram.unstage()
                           

                        with dpg.tab(label="Tyre temperatures"):
                            histogram = util.UiHistogram.create("tyre_temp_histogram", u"Tyre temperature, [\N{DEGREE SIGN}C]", "Proportion")
                            Variables.get_var("graphs")["tyre_temp_histogram"] = histogram
                            histogram.unstage()

                        with dpg.tab(label="Section report", tag="tyre_report"):
                            dpg.add_button(label="Produce Report", callback=CallbackManager.get_callback("optimise_pressure"))
                            


                with dpg.tab(label="Dampers"):
                    with dpg.tab_bar():
                        with dpg.tab(label="Damper motion histogram"):
                            histogram = util.UiHistogram.create("damper_histogram", "Motion, [mm/s]", "Proportion")
                            Variables.get_var("graphs")["damper_histogram"] = histogram
                            histogram.unstage()
                            dpg.add_button(label="optimise_dampers", callback=CallbackManager.get_callback("optimise_dampers"))

                        with dpg.tab(label="Section report", tag="damper_report"):
                            dpg.add_button(label="Produce Report", callback=CallbackManager.get_callback("optimise_dampers"))

                with dpg.tab(label="Graph"):
                    with dpg.tab_bar():
                        with dpg.tab(label="Graph"):
                            with dpg.group(horizontal=True, tag="graph_group_h"):
                                with dpg.tree_node(label="Motec channels", tag="motec_channels_list"):
                                    pass

                                with dpg.group(tag="graph_group"):        
                                    plot = util.UiPlot("telemetry", 13, "Telemetry", "Distance, [m]")    
                                    plot.unstage(parent="graph_group")
                                    Variables.set_var("plot", plot)


                                #create plots for every unit of measurement
                with dpg.tab(label="Engineer", tag="engineer"):
                     with dpg.tab_bar():
                        with dpg.tab(label="Corner Performance Table", tag="Corner Performance Table"):
                            dpg.add_button(label="Refresh", tag="refresh_corner_performance_table_button", callback=CallbackManager.get_callback("refresh_corner_performance_table_button"))
                            # dpg.add_button(label="corner_analysis", callback=CallbackManager.get_callback("corner_analysis"))
                        with dpg.tab(label="Engineer Report", tag="Engineer Report"):
                            dpg.add_button(label="Produce Report", tag="engineer_report_button", callback=CallbackManager.get_callback("engineer_report"))



            dpg.bind_item_handler_registry("telemetry_window", "widget handler")
            #with dpg.theme() as item_theme:
            #    with dpg.theme_component(dpg.mvAll):
            #        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (200, 200, 100), category=dpg.mvThemeCat_Core)
            #        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0, category=dpg.mvThemeCat_Core)

    @staticmethod
    def telemetry_window_update_tyre_report():
        telemetry = Variables.get_var("telemetry")
        track = Variables.get_var("track")
        lap = Variables.get_var("selected_lap")
        tyre_optimiser = Variables.get_var("tyre_optimiser")
        lap_stint = (lap, 1) if lap is not None else (0, 1)
        if telemetry == None:
            print("Error")
            return
        telemetry = telemetry["data"]
        pressures = telemetry.analyse_tyre_press(lap_stint, track)
        actions = tyre_optimiser.infer(pressures)


        dpg.push_container_stack("tyre_report")
        
        children = dpg.get_item_children("tyre_report", 1)
        for child in children:
            if dpg.get_item_alias(child) in ["tyre_table", "apply_changes_tyres_button", "store_results_tyres_button"]:
                dpg.delete_item(child)
       
            



        
        with dpg.table(resizable=True, policy=dpg.mvTable_SizingStretchProp, width=1500,
                            borders_outerH=True, borders_innerV=True, borders_outerV=True, header_row=True, tag="tyre_table"):
            dpg.add_table_column(label="Tyre", width_stretch=True, init_width_or_weight=1/7)
            dpg.add_table_column(label="Current Corner Pressure, psi", width_stretch=True, init_width_or_weight=1/7)
            dpg.add_table_column(label="Current Average Pressure, psi", width_stretch=True, init_width_or_weight=1/7)
            dpg.add_table_column(label="Desired Pressure", width_stretch=True, init_width_or_weight=1/7)
            dpg.add_table_column(label="Predicted change per click", width_stretch=True, init_width_or_weight=1/7)
            dpg.add_table_column(label="Recommended change, clicks", width_stretch=True, init_width_or_weight=1/7)
            dpg.add_table_column(label="Select change, clicks", width_stretch=True, init_width_or_weight=1/7)
                         
                         
            names = {
                "lf": "Left Front",
                "rf": "Right Front",
                "lr": "Left Rear",
                "rr": "Right Rear",
            }                               
            #corners
            #lf
            slider_ids = {}
            for tyre in ["lf", "rf", "lr", "rr"]:
                with dpg.table_row():
                    dpg.add_text(names[tyre])
                    dpg.add_text(pressures[tyre][1])
                    dpg.add_text(pressures[tyre][3])
                    dpg.add_text("27.5-27.7")
                    dpg.add_text(actions[tyre][0])
                    dpg.add_text(actions[tyre][1])
                    uid = dpg.add_input_int(default_value=actions[tyre][1])
                    slider_ids[tyre] = uid

        dpg.add_button(label="Apply changes", callback=CallbackManager.get_callback("apply_changes_tyres"), user_data = {"pressures":pressures, "slider_ids": slider_ids}, tag="apply_changes_tyres_button")
        dpg.add_button(label="Store results", callback=CallbackManager.get_callback("store_results_tyres"), user_data = {"pressures":pressures}, tag="store_results_tyres_button")


        
        dpg.pop_container_stack()


    @staticmethod
    def telemetry_window_update_damper_report():
        telemetry = Variables.get_var("telemetry")
        track = Variables.get_var("track")
        lap = Variables.get_var("selected_lap")
        tyre_optimiser = Variables.get_var("tyre_optimiser")
        lap_stint = (lap, 1) if lap is not None else (0, 1)
        if telemetry == None:
            print("Error")
            return
        telemetry = telemetry["data"]
        damper_optimiser = Variables.get_var("damper_optimiser")
        dampers = telemetry.analyse_damper_histogram(lap_stint)

        actions = damper_optimiser.infer(dampers)


        dpg.push_container_stack("damper_report")
        
        children = dpg.get_item_children("damper_report", 1)
        for child in children:
            if dpg.get_item_alias(child) in ["damper_table"]:
                dpg.delete_item(child)


        with dpg.table(resizable=True, policy=dpg.mvTable_SizingStretchProp, width=1500,
                            borders_outerH=True, borders_innerV=True, borders_outerV=True, header_row=True, tag="damper_table"):
            dpg.add_table_column(label="Damper", width_stretch=True, init_width_or_weight=1/6)
            dpg.add_table_column(label="Spring Rate", width_stretch=True, init_width_or_weight=1/6)
            dpg.add_table_column(label="Slow Bump", width_stretch=True, init_width_or_weight=1/6)
            dpg.add_table_column(label="Slow Rebound", width_stretch=True, init_width_or_weight=1/6)
            dpg.add_table_column(label="Fast Bump", width_stretch=True, init_width_or_weight=1/6)
            dpg.add_table_column(label="Fast Rebound", width_stretch=True, init_width_or_weight=1/6)
                         
                         
            names = {
                "lf": "Left Front",
                "rf": "Right Front",
                "lr": "Left Rear",
                "rr": "Right Rear",
            }                               
            #corners
            #lf
            slider_ids = {}
            for tyre in ["lf", "rf", "lr", "rr"]:
                with dpg.table_row():
                    dpg.add_text(names[tyre])
                    dpg.add_text(f'{actions[tyre]["SPRING_RATE"]:.2f}')
                    dpg.add_text(f'{actions[tyre]["SLOW_BUMP"]:.2f}')
                    dpg.add_text(f'{actions[tyre]["SLOW_REBOUND"]:.2f}')
                    dpg.add_text(f'{actions[tyre]["FAST_BUMP"]:.2f}')
                    dpg.add_text(f'{actions[tyre]["FAST_REBOUND"]:.2f}')
        
        dpg.pop_container_stack()


    @staticmethod
    def telemetry_window_load_telemetry():
        Variables.set_var("selected_lap", None)
        telemetry = Variables.get_var("telemetry")
        if telemetry == None:
            print("Error")
            return
        dpg.set_value("telemetry_window_telemetry_file_name_text", "Telemetry_file: " + telemetry["name"])

        # create laptable

        dpg.push_container_stack("lap_table_tab")
        
        children = dpg.get_item_children("lap_table_tab", 1)
        for child in children:
            if dpg.get_item_alias(child) in ["laptable"]:
                dpg.delete_item(child)

        
        with dpg.table(resizable=True, policy=dpg.mvTable_SizingStretchProp, width=1900,
                            borders_outerH=True, borders_innerV=True, borders_outerV=True, header_row=True, tag="laptable"):
            dpg.add_table_column(label="Lap number", width_stretch=True, init_width_or_weight=1/3)
            dpg.add_table_column(label="Lap time", width_stretch=True, init_width_or_weight=1/3)
            dpg.add_table_column(label="Select lap", width_stretch=True, init_width_or_weight=1/3)
                                                        

            lap_times = telemetry["data"].laps_times
            lap_select = CheckboxGroup.create("lap_select")
            for idx, val in enumerate(lap_times):
                with dpg.table_row():
                    dpg.add_text(idx)
                    dpg.add_text(f"{str(int(val)//60)}:{str(int(val)%60).zfill(2)}:{str(int(val*1000)%1000).zfill(3)}")
                    rb = dpg.add_checkbox(callback=CallbackManager.get_callback("select_checkbox"), user_data={"group":"lap_select", "value":idx})
                    lap_select.add_checkbox(rb)

        #dpg.move_item("laptable", before="tb1")
        dpg.pop_container_stack()


        units = telemetry["data"].get_unit_map().items()
        plot = Variables.get_var("plot")
        Variables.set_var("motec_channels", {})

        children = dpg.get_item_children("graph_group_h", 1)
        for child in children:
            if dpg.get_item_alias(child) in ["motec_channels_list"]:
                for c in dpg.get_item_children(child, 1):
                    dpg.delete_item(c)

        for unit, cs in units:
            plot.add_subplot(unit, f'[{unit}]')
            for name in cs:
                dpg.push_container_stack("motec_channels_list")
                dpg.add_selectable(label=name, callback=CallbackManager.get_callback("select_motec_channel"), tag=name+"_tag", user_data={"chan_name": name})
                dpg.pop_container_stack()
                Variables.get_var("motec_channels")[name] = False
                plot.add_chan(unit, name, name)
        

        WindowManager.update_telemerty_graphs()


        
        # set lap select mode

        print(dpg.get_item_children("telemetry_window", 1))


    @staticmethod
    def update_telemerty_graphs():
        lap = Variables.get_var("selected_lap")
        if lap == None:
            lap_stint = None
        else:
            lap_stint = (lap, 1)

        telemetry = Variables.get_var("telemetry")["data"]
            
        # tyre pressure histogram
        data = telemetry.get_tyre_press_histogram(lap_stint=lap_stint)
        hist = Variables.get_var("graphs")["tyre_press_histogram"]
        hist.set_resolution(data["resolution"])
        hist.update_histogram(
            data["min_scale_x"], 
            data["max_scale_x"], 
            0, 
            data["max_scale_y"], 
            (
                (data["lf"]["bin_centers"], data["lf"]["data"]), 
                (data["rf"]["bin_centers"], data["rf"]["data"]), 
                (data["lr"]["bin_centers"], data["lr"]["data"]), 
                (data["rr"]["bin_centers"], data["rr"]["data"])
            )
        )

        # damper histogram
        data = telemetry.get_damper_histogram(lap_stint=lap_stint, res=8, low=-204, high=204)
        telemetry.analyse_damper_histogram(lap_stint=lap_stint)
        hist = Variables.get_var("graphs")["damper_histogram"]
        hist.set_resolution(data["resolution"])
        hist.update_histogram(
            data["min_scale_x"], 
            data["max_scale_x"], 
            0, 
            data["max_scale_y"], 
            (
                (data["lf"]["bin_centers"], data["lf"]["data"]), 
                (data["rf"]["bin_centers"], data["rf"]["data"]), 
                (data["lr"]["bin_centers"], data["lr"]["data"]), 
                (data["rr"]["bin_centers"], data["rr"]["data"])
            )
        )

        # tyre temp histogram
        data = telemetry.get_tyre_temp_histogram(lap_stint=lap_stint)
        hist = Variables.get_var("graphs")["tyre_temp_histogram"]
        hist.set_resolution(data["resolution"])
        hist.update_histogram(
            data["min_scale_x"], 
            data["max_scale_x"], 
            0, 
            data["max_scale_y"], 
            (
                (data["lf"]["bin_centers"], data["lf"]["data"]), 
                (data["rf"]["bin_centers"], data["rf"]["data"]), 
                (data["lr"]["bin_centers"], data["lr"]["data"]), 
                (data["rr"]["bin_centers"], data["rr"]["data"])
            )
        )

        data = telemetry.get_chan_plot("speedkmh")

        # plot selected channs
        chan_selected = Variables.get_var("motec_channels")
        plot = Variables.get_var("plot")
        for name, selected in chan_selected.items():
            if selected:
                data = telemetry.get_chan_plot(name, lap_stint=lap_stint)
                unit = telemetry.get_chan_unit(name)
                plot.update_chan_value(name, unit, data["dist"], data["data"])
            else:
                unit = telemetry.get_chan_unit(name)
                plot.update_chan_value(name, unit, (), ())



    @staticmethod
    def telemetry_window_load_setup():
        setup = Variables.get_var("setup")
        if setup == None:
            print("Error")
            return
        dpg.set_value("telemetry_window_setup_file_name_text", "Setup_file: " + setup["name"])
        print(setup["setup"].data)

    @staticmethod
    def analyse_telemetry():
        lap = Variables.get_var("selected_lap")
        if lap == None:
            lap_stint = None
        else:
            lap_stint = (lap, 1)

        track = Variables.get_var("track")
        telemetry = Variables.get_var("telemetry")["data"]

        telemetry.analyse_g_lat(lap_stint, track)
        telemetry.analyse_tc(lap_stint, track)
        telemetry.analyse_tyre_press(lap_stint, track)

        WindowManager.display_lap_setion_table()

    @staticmethod
    def display_lap_setion_table():
        dpg.push_container_stack("Corner Performance Table")
        
        children = dpg.get_item_children("Corner Performance Table", 1)
        for child in children:
            if dpg.get_item_alias(child) in ["lap_section_table"]:
                for i in dpg.get_item_children(child, 1):
                    print(dpg.get_item_alias(i))
                dpg.delete_item(child)

        

        with dpg.table(resizable=True, policy=dpg.mvTable_SizingStretchProp, width=1880,
                            borders_outerH=True, borders_innerV=True, borders_outerV=True, header_row=True, tag="lap_section_table"):
            dpg.add_table_column(label="Lap section", width_stretch=True, init_width_or_weight=1/15)
            dpg.add_table_column(label="Type", width_stretch=True, init_width_or_weight=2/15)
            dpg.add_table_column(label="Driver's feedback entry", width_stretch=True, init_width_or_weight=4/15)
            dpg.add_table_column(label="Driver's feedback mid", width_stretch=True, init_width_or_weight=4/15)
            dpg.add_table_column(label="Driver's feedback exit", width_stretch=True, init_width_or_weight=4/15)
                                                        
            track = Variables.get_var("track")
            section_names = track.section_names
            section_types = track.section_types
            lap = Variables.get_var("selected_lap")
            telemetry = Variables.get_var("telemetry")["data"]

            if lap == None:
                lap = 0

            lap_stint = (lap, 1)
            lap_time = telemetry.laps_times[lap]

            if Variables.get_var("config")["feedback_mode"] == 1:
                performance_array = telemetry.analyse_oversteer_understeer(lap_stint, track)
                profile = telemetry.get_stint_profile(performance_array, lap_stint, track)
                names = ["slow", "medium", "fast"]
                types = [ldparser.SectionType.CORNER_SLOW, ldparser.SectionType.CORNER_MED, ldparser.SectionType.CORNER_FAST]
                for idx in range(3):
                    with dpg.table_row():
                        dpg.add_text("Corner")
                        dpg.add_text(types[idx].description())
                        
                        dpg.add_slider_float(label="understeer - oversteer", min_value=-10, max_value=10, default_value=profile[3*idx],   format="%f", tag=f"corner_{names[idx]}_entry")
                        dpg.add_slider_float(label="understeer - oversteer", min_value=-10, max_value=10, default_value=profile[3*idx+1], format="%f", tag=f"corner_{names[idx]}_mid")
                        dpg.add_slider_float(label="understeer - oversteer", min_value=-10, max_value=10, default_value=profile[3*idx+2], format="%f", tag=f"corner_{names[idx]}_exit")


            elif Variables.get_var("config")["feedback_mode"] == 2:
                performance_array = telemetry.analyse_oversteer_understeer(lap_stint, track)

                counter = 0
                for idx in range(len(section_names)):
                    if section_types[idx] == ldparser.SectionType.STRAIGHT: continue
                    with dpg.table_row():
                        dpg.add_text(section_names[idx])
                        dpg.add_text(section_types[idx].description())
                        
                        profile = performance_array[counter]
                        counter += 1
                        dpg.add_slider_float(label="oversteer - understeer", min_value=-10, max_value=10, default_value=profile[0], format="%f", tag=f"corner_{counter}_entry")
                        dpg.add_slider_float(label="oversteer - understeer", min_value=-10, max_value=10, default_value=profile[1], format="%f", tag=f"corner_{counter}_mid")
                        dpg.add_slider_float(label="oversteer - understeer", min_value=-10, max_value=10, default_value=profile[2], format="%f", tag=f"corner_{counter}_exit")

                Variables.set_var("corner_count", counter)

        dpg.move_item("lap_section_table", before="refresh_corner_performance_table_button")
        dpg.pop_container_stack()

    @staticmethod
    def update_last_state_table():
        dpg.push_container_stack("Engineer Report")
        
        children = dpg.get_item_children("Engineer Report", 1)
        for child in children:
            if dpg.get_item_alias(child) in ["nn_learn_table"]:
                dpg.delete_item(child)

        with dpg.table(resizable=True, policy=dpg.mvTable_SizingStretchProp, width=1900,
                            borders_outerH=True, borders_innerV=True, borders_outerV=True, header_row=True, tag="nn_learn_table"):
            dpg.add_table_column(label="Last State", width_stretch=True, init_width_or_weight=1/5)
            dpg.add_table_column(label="Lap Time", width_stretch=True, init_width_or_weight=1/5)
            dpg.add_table_column(label="Profile", width_stretch=True, init_width_or_weight=1/5)
            dpg.add_table_column(label="Chosen Action", width_stretch=True, init_width_or_weight=1/5)
            dpg.add_table_column(label="Learn", width_stretch=True, init_width_or_weight=1/5)
            



            last_state = Variables.get_var("last_state")
            current_state = Variables.get_var("current_state")
            for state in [last_state, current_state]:
                with dpg.table_row():
                    dpg.add_text(state["telemetry_name"] if state is not None else "-")

                    dpg.add_text(f"{str(int(state['lap_time'])//60)}:{str(int(state['lap_time'])%60).zfill(2)}:{str(int(state['lap_time']*1000)%1000).zfill(3)}" if state is not None else "-")

                    description = [
                        "Slow_corner_entry", 
                        "Slow_corner_middle", 
                        "Slow_corner_exit", 
                        "Medium_corner_entry", 
                        "Medium_corner_middle", 
                        "Medium_corner_exit", 
                        "Slow_corner_entry", 
                        "Slow_corner_middle", 
                        "Slow_corner_exit", 
                    ]
                    str_profile = [f'{name}: {val:.2f}' for name, val in zip(description, state["profile"])] if state is not None else []
                    dpg.add_text("\n".join(str_profile) if state is not None else "-")
                    

                    if state is last_state:
                        dpg.add_text(state["chosen_action"] if state is not None else "-")
                        if not (last_state is None or current_state is None or last_state["telemetry_name"] == current_state["telemetry_name"]):
                            Variables.set_var("nn_learn_btn", [dpg.add_button(label="Learn", callback=CallbackManager.get_callback("nn_learn"))])

        dpg.move_item("nn_learn_table", before="engineer_report_table")
        dpg.pop_container_stack()


    @staticmethod
    def engineer_report():
        agent = Variables.get_var("agent")
        setup = Variables.get_var("setup")
        inference = Variables.get_var("inference")
        telemetry = Variables.get_var("telemetry")
        telemetry_name = telemetry["name"]
        telemetry = telemetry["data"]
        track = Variables.get_var("track")
        lap = Variables.get_var("selected_lap")
        if lap == None:
            return
        else:
            lap_stint = (lap, 1)
        lap_time = telemetry.laps_times[lap]

        if Variables.get_var("config")["feedback_mode"] == 1:
            steer_profile = []
            for i in ["slow", "medium", "fast"]:
                steer_profile.extend((dpg.get_value(f"corner_{i}_entry"), dpg.get_value(f"corner_{i}_mid"), dpg.get_value(f"corner_{i}_exit")))

            steer_profile = np.array(steer_profile)

        elif Variables.get_var("config")["feedback_mode"] == 2:
            corner_array = []
            for i in range(1, Variables.get_var("corner_count") + 1):
                corner_array.append((dpg.get_value(f"corner_{i}_entry"), dpg.get_value(f"corner_{i}_mid"), dpg.get_value(f"corner_{i}_exit")))
            
            steer_profile = telemetry.get_stint_profile(corner_array, lap_stint, track)

        current_state = {
            "telemetry_name": telemetry_name,
            "lap_time": lap_time,
            "profile": steer_profile,
        }       
        Variables.set_var("current_state", current_state)

        inference_system_results = inference.inference(steer_profile, setup["setup"].normalized_data)
        nn_results = agent.calculate_actions(steer_profile)
        possible_actions = setup["setup"].possible_actions

        dpg.push_container_stack("Engineer Report")
        
        children = dpg.get_item_children("Engineer Report", 1)
        for child in children:
            if dpg.get_item_alias(child) in ["engineer_report_table", "nn_learn_table"]:
                dpg.delete_item(child)

        with dpg.table(resizable=True, policy=dpg.mvTable_SizingStretchProp, width=1900,
                            borders_outerH=True, borders_innerV=True, borders_outerV=True, header_row=True, tag="nn_learn_table"):
            dpg.add_table_column(label="State", width_stretch=True, init_width_or_weight=1/5)
            dpg.add_table_column(label="Lap Time", width_stretch=True, init_width_or_weight=1/5)
            dpg.add_table_column(label="Profile", width_stretch=True, init_width_or_weight=1/5)
            dpg.add_table_column(label="Chosen Action", width_stretch=True, init_width_or_weight=1/5)
            dpg.add_table_column(label="Learn", width_stretch=True, init_width_or_weight=1/5)
            

            last_state = Variables.get_var("last_state")

            for state in [last_state, current_state]:
                with dpg.table_row():
                    dpg.add_text(state["telemetry_name"] if state is not None else "-")

                    dpg.add_text(f"{str(int(state['lap_time'])//60)}:{str(int(state['lap_time'])%60).zfill(2)}:{str(int(state['lap_time']*1000)%1000).zfill(3)}" if state is not None else "-")

                    description = [
                        "Slow_corner_entry", 
                        "Slow_corner_middle", 
                        "Slow_corner_exit", 
                        "Medium_corner_entry", 
                        "Medium_corner_middle", 
                        "Medium_corner_exit", 
                        "Fast_corner_entry", 
                        "Fast_corner_middle", 
                        "Fast_corner_exit", 
                    ]
                    str_profile = [f'{name}: {val:.2f}' for name, val in zip(description, state["profile"])] if state is not None else []
                    dpg.add_text("\n".join(str_profile) if state is not None else "-")
                    

                    if state is last_state:
                        dpg.add_text(state["chosen_action"] if state is not None else "-")
                        if not (last_state is None or current_state is None or last_state["telemetry_name"] == current_state["telemetry_name"]):
                            Variables.set_var("nn_learn_btn", [dpg.add_button(label="Learn", callback=CallbackManager.get_callback("nn_learn"))])


                    

        with dpg.table(resizable=True, policy=dpg.mvTable_SizingStretchProp, width=1900,
                            borders_outerH=True, borders_innerV=True, borders_outerV=True, header_row=True, tag="engineer_report_table"):
            dpg.add_table_column(label="Action", width_stretch=True, init_width_or_weight=1/5)
            dpg.add_table_column(label="Neural Network Evaluation", width_stretch=True, init_width_or_weight=1/5)
            dpg.add_table_column(label="Inference System Evaluation", width_stretch=True, init_width_or_weight=1/5)
            dpg.add_table_column(label="", width_stretch=True, init_width_or_weight=1/5)
            dpg.add_table_column(label="Choose", width_stretch=True, init_width_or_weight=1/5)
                         
            btns = []        
            
            for action in setuploader.Actions:
                with dpg.table_row():
                    if action not in possible_actions: continue

                    dpg.add_text(action.name)
                    dpg.add_text(f'{nn_results[action]:.2f}')
                    dpg.add_text(f'{inference_system_results[action.name]:.2f}')
                    dpg.add_text("")
                    btn_id = dpg.add_button(label="Choose", callback=CallbackManager.get_callback("nn_choose_action"), user_data={
                        "telemetry_name": telemetry_name,
                        "lap_time": lap_time,
                        "profile": steer_profile,
                        "chosen_action": action
                    })
                    if last_state is not None and last_state["telemetry_name"] == telemetry_name:
                        dpg.configure_item(btn_id, enabled=False, show=False)
                    btns.append(btn_id)

            Variables.set_var("nn_choose_action_btns", btns)       
        dpg.pop_container_stack()

if __name__ == "__main__":
    pass
