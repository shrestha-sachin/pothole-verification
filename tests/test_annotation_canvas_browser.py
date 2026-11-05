"""Optional Chromium interaction checks using a synthetic image, never user footage.
Run after installing playwright and its Chromium browser.
"""
import base64
import functools
import io
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest
from PIL import Image


def test_canvas_draw_select_resize_zoom_undo_and_button_hit_areas():
    playwright = pytest.importorskip('playwright.sync_api')
    directory=Path('dashboard/annotation_canvas').resolve()
    handler=functools.partial(SimpleHTTPRequestHandler,directory=str(directory))
    server=ThreadingHTTPServer(('127.0.0.1',0),handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    image=Image.new('RGB',(1000,600),'#64748b');output=io.BytesIO();image.save(output,format='PNG')
    data='data:image/png;base64,'+base64.b64encode(output.getvalue()).decode()
    try:
        with playwright.sync_playwright() as p:
            try: browser=p.chromium.launch(headless=True)
            except playwright.Error as exc: pytest.skip(f'Chromium unavailable: {exc}')
            page=browser.new_page(viewport={'width':1200,'height':900})
            errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html')
            page.evaluate('''(image) => {
              window.edits=[];
              window.args={image_id:'fixture',image,width:1000,height:600,boxes:[],revision:0};
              window.addEventListener('message',event=>{
                if(event.data.type==='streamlit:setComponentValue'){
                  window.edits.push(event.data.value);
                  window.args.revision=event.data.value.revision;
                  window.args.boxes=event.data.value.boxes;
                  window.postMessage({type:'streamlit:render',args:window.args},'*');
                }
              });
              window.postMessage({type:'streamlit:render',args:window.args},'*');
            }''',data)
            page.wait_for_function("document.getElementById('status').textContent.includes('Drag to draw')")
            area=page.locator('#canvas').bounding_box()
            # Draw backwards to verify normalization. Pointermove must not emit server edits.
            x,y=area['x'],area['y']
            page.mouse.move(x+650,y+350);page.mouse.down();page.mouse.move(x+400,y+200,steps=15)
            assert page.evaluate('edits.length')==0
            page.mouse.up();page.wait_for_function('edits.length===1')
            first=page.evaluate('edits[0].boxes[0]')
            assert first['w']>0 and first['h']>0
            page.locator('#select').click()
            assert page.locator('#select').get_attribute('aria-pressed')=='true'
            page.mouse.move(x+500,y+260);page.mouse.down();page.mouse.move(x+540,y+280,steps=10);page.mouse.up()
            page.wait_for_function('edits.length===2')
            assert page.evaluate('edits[1].boxes[0].x')>first['x']
            # Real resize by dragging the current box's southeast handle.
            box=page.evaluate('edits[1].boxes[0]')
            z=min((area['width']-32)/1000,(area['height']-32)/600)
            tx=(area['width']-1000*z)/2;ty=(area['height']-600*z)/2
            corner_x=x+tx+(box['x']+box['w'])*z
            corner_y=y+ty+(box['y']+box['h'])*z
            page.mouse.move(corner_x,corner_y);page.mouse.down()
            page.mouse.move(corner_x+30,corner_y+20,steps=10);page.mouse.up()
            page.wait_for_function('edits.length===3')
            assert page.evaluate('edits[2].boxes[0].w')>box['w']
            page.locator('#selectedCategory').select_option('shadow')
            page.wait_for_function('edits.length===4')
            assert page.evaluate('edits[3].boxes[0].category')=='shadow'
            page.locator('#undo').click();page.wait_for_function('edits.length===5')
            assert page.evaluate('edits[4].boxes[0].category')=='pothole'
            page.locator('#redo').click();page.wait_for_function('edits.length===6')
            page.locator('#select').click();page.mouse.click(x+540,y+280)
            page.locator('#delete').click();page.wait_for_function('edits.length===7')
            assert page.evaluate('edits[6].boxes.length')==0
            before=page.evaluate('edits.length');page.locator('#in').click();page.locator('#fit').click()
            assert page.evaluate('edits.length')==before
            assert not errors
            browser.close()
    finally:
        server.shutdown();server.server_close()
