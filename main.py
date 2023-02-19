import dearpygui.dearpygui as dpg
import dearpygui.demo as demo

import src.ui.Ui as ui
import src.ldparser.q_learning as q_learning

data = [0 ,1, 2]

agent = q_learning.Agent(gamma=0.99, epsilon = 1.0, batch_size = 64, n_outputs = 1, eps_end = 0.01,
			input_dims=[10], lr=0.01)

def initialise():
	ui.Variables.initialise_vars()
	ui.CallbackManager.initialise_callbacks()
	ui.WindowManager.initialise_windows()
	ui.WindowManager.set_main_window("main_window")
#creating context
dpg.create_context()

dpg.create_viewport(title='Custom Title', width=1200, height=800)
dpg.set_viewport_vsync(True)
dpg.setup_dearpygui()
dpg.show_viewport()
#dpg.toggle_viewport_fullscreen()
#demo.show_demo()
# initialising custom UI callbacks and windows

initialise()

# print children
#print(dpg.get_item_children(dpg.last_root()))

# print children in slot 1
#print(dpg.get_item_configuration("test1"))

# check draw_line's slot
#print(dpg.get_item_slot(dpg.last_item()))

dpg.start_dearpygui()
dpg.destroy_context()