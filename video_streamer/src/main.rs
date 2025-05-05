use once_cell::sync::Lazy;
use opencv::core::Point;
use opencv::prelude::*;
use opencv::{highgui, videoio, Result};
use std::sync::Mutex;
use std::time::Instant;

const WAIT_MS: i32 = 10;
const ESC_KEY: i32 = 27;

// Globally thread-safe containers that are initialized on first access
static STATE: Lazy<Mutex<State>> = Lazy::new(|| Mutex::new(State::new()));
static ACTIONS: Lazy<Mutex<Vec<MouseAction>>> = Lazy::new(|| Mutex::new(Vec::new()));

// Represent prompts from Segment Anything 2 (https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)
// Prompts are positive/negative clicks, boxes, or masks
#[derive(Debug)]
enum MouseAction {
    SingleClick(Point), // Indicate positive click
    DoubleClick(Point), // Indicate negative click
    DrawBox(Point, Point),
}

struct State {
    start: Option<Point>,
    last_click: Option<(Point, Instant)>,
    double_click_interval_ms: u128,
    click_radius: i32,
    drag_threshold: i32,
}

impl State {
    fn new() -> Self {
        State {
            start: None,
            last_click: None,
            double_click_interval_ms: 500,
            click_radius: 5 * 5,
            drag_threshold: 5 * 5,
        }
    }

    fn flush_pending(&mut self, actions: &mut Vec<MouseAction>) {
        if let Some((pt, t0)) = self.last_click {
            if Instant::now().duration_since(t0).as_millis() > self.double_click_interval_ms {
                actions.push(MouseAction::SingleClick(pt));
                self.last_click = None;
            }
        }
    }

    fn handle_event(&mut self, event: i32, x: i32, y: i32, actions: &mut Vec<MouseAction>) {
        // When the first click event occurs via LBUTTONUP, we don't immediately emit SingleClick.
        // We store it in `st.last_click` as a pending click, along with a timestamp.
        // If that pending click has expired its double-click window, we now emit it as a true SingleClick
        // and clear the pending slot.
        // If it's still within the double-click window, we leave it pending, hoping a second click comes.
        //
        // **SingleClick => first click occurs and it's been more than the double-click window
        // **DoubleClick => first click followed by second click within `double_click_interval_ms` and two points are within `click_radius`
        // **DrawBox => distance between two points has exceeded `drag_threshold`
        let now = Instant::now();
        let pt = Point::new(x, y);

        match event {
            // Record potential drag start
            highgui::EVENT_LBUTTONDOWN => {
                self.flush_pending(actions);
                self.start = Some(pt);
            }

            // Release: decide drag vs single vs double-click
            highgui::EVENT_LBUTTONUP => {
                if let Some(p0) = self.start.take() {
                    let dx = x - p0.x;
                    let dy = y - p0.y;
                    let dist = dx * dx + dy * dy;

                    if dist > self.drag_threshold {
                        // DRAG: draw box
                        actions.push(MouseAction::DrawBox(p0, pt));
                    } else {
                        // CLICK: check for pending click to form a double
                        if let Some((last_pt, last_t)) = self.last_click {
                            let dt = now.duration_since(last_t).as_millis();

                            // Calculate squared distance between two points
                            let dr = (pt.x - last_pt.x).pow(2) + (pt.y - last_pt.y).pow(2);
                            if dt <= self.double_click_interval_ms && dr <= self.click_radius {
                                actions.push(MouseAction::DoubleClick(pt));
                                self.last_click = None;
                                return;
                            }
                        }

                        // Buffer this as a pending single-click
                        self.last_click = Some((pt, now));
                    }
                }
            }
            _ => {}
        }
    }
}

fn pretty_print(actions: &mut Vec<MouseAction>) {
    // Empty the buffer so we don't re-print old events on the next frame
    for act in actions.drain(..) {
        match act {
            MouseAction::SingleClick(pt) => {
                println!("Single click at {:?}", pt);
            }
            MouseAction::DoubleClick(pt) => {
                println!("Double click at {:?}", pt);
            }
            MouseAction::DrawBox(p1, p2) => {
                println!("Draw box from {:?} to {:?}", p1, p2);
            }
        }
    }
}

fn with_context<R>(f: impl FnOnce(&mut State, &mut Vec<MouseAction>) -> R) -> R {
    let mut st = STATE.lock().unwrap();
    let mut actions = ACTIONS.lock().unwrap();
    f(&mut st, &mut actions)
}

fn main() -> Result<()> {
    let window = "Video Capture";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?; // 0 is the default camera
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    let mouse_callback = Box::new(move |event: i32, x: i32, y: i32, _flags: i32| {
        with_context(|st, actions| {
            st.handle_event(event, x, y, actions);
        })
    });
    highgui::set_mouse_callback(window, Some(mouse_callback))?;

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        if frame.empty() {
            break;
        }

        highgui::imshow(window, &frame)?;

        if highgui::wait_key(WAIT_MS)? == ESC_KEY {
            break;
        }

        with_context(|st, actions| {
            // Flushing a stale pending click ensures when no second click happens in time,
            // the main loop notices the timeout, emits one SingleClick, and clears the pending.
            st.flush_pending(actions);
            pretty_print(actions);
            actions.clear();
        });
    }
    Ok(())
}
