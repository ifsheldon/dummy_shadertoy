use pixel_canvas::canvas::CanvasInfo;
use pixel_canvas::input::{Event, WindowEvent};
use pixel_canvas::input::glutin::event::{VirtualKeyCode, ElementState};

pub struct KeyboardMouseStates {
    pub received_char: bool,
    pub received_mouse_press: bool,
    pub received_keycode: bool,
    pub keycode: VirtualKeyCode,
    pub char: char,
    pub x: i32,
    pub y: i32,
    pub virtual_x: i32,
    pub virtual_y: i32
}

impl KeyboardMouseStates {
    pub fn new() -> Self
    {
        Self {
            received_char: false,
            received_mouse_press: false,
            received_keycode: false,
            keycode: VirtualKeyCode::Escape,
            char: ' ',
            x: 0,
            y: 0,
            virtual_x: 0,
            virtual_y: 0
        }
    }

    fn reset_flags(&mut self)
    {
        self.received_keycode = false;
        self.received_char = false;
        self.received_mouse_press = false;
    }

    pub fn handle_input(info: &CanvasInfo, state_to_change: &mut KeyboardMouseStates, event: &Event<()>) -> bool
    {
        state_to_change.reset_flags();
        match event {
            Event::WindowEvent {
                window_id: _window_id, event
            } => {
                match event {
                    WindowEvent::CursorMoved { device_id: _device_id, position, modifiers: _modifiers } => {
                        let (x, y) :(i32, i32) = (*position).into();
                        state_to_change.virtual_x = x;
                        state_to_change.virtual_y = y;
                        state_to_change.x = (x as f64 * info.dpi) as i32;
                        state_to_change.y = ((info.height as i32 - y) as f64 * info.dpi) as i32;
                        true
                    }
                    WindowEvent::ReceivedCharacter(c) => {
                        state_to_change.received_char = true;
                        state_to_change.char = *c;
                        true
                    }
                    WindowEvent::KeyboardInput { device_id: _device_id, input, is_synthetic: _is_synthetic } => {
                        match input.virtual_keycode {
                            None => false,
                            Some(keycode) => {
                                state_to_change.received_keycode = true;
                                state_to_change.keycode = keycode;
                                true
                            }
                        }
                    }
                    WindowEvent::MouseInput { device_id: _device_id, state, button, modifiers: _modifiers } => {
                        match state
                        {
                            ElementState::Pressed => {
                                state_to_change.received_mouse_press = true;
                                return true;
                            }
                            ElementState::Released => false
                        }
                    }
                    // WindowEvent::CursorEntered { .. } => {true}
                    // WindowEvent::CursorLeft { .. } => {}
                    // WindowEvent::MouseWheel { .. } => {true}
                    _ => { false }
                }
            },
            _ => false
        }
    }
}
