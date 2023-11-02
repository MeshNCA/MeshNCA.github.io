import {MeshNCA} from './meshNCA.js'
import {Mesh} from './mesh_utils.js'

function isInViewport(element) {
    var rect = element.getBoundingClientRect();
    var html = document.documentElement;
    var w = window.innerWidth || html.clientWidth;
    var h = window.innerHeight || html.clientHeight;
    return rect.top < h && rect.left < w && rect.bottom > 0 && rect.right > 0;
}

export function createDemo(divId) {
    const root = document.getElementById(divId);
    const $ = q => root.querySelector(q);
    const $$ = q => root.querySelectorAll(q);

    const speed_sets = [1, 2, 4, 8, 16];
    let W = 512;
    // W = Math.min(W, window.screen.width);
    let H = W;
    // alert(window.screen.width);
    let ca = null;
    let paused = false;

    const clearColor = [0.90, 0.90, 0.90, 1.0];
    // const clearColor = [1, 1, 1, 1.0];
    const canvas = $('#demo-canvas');
    canvas.width = W;
    canvas.height = H;
    let gl = canvas.getContext("webgl2", {antialias: true, preserveDrawingBuffer: true});


    if (!gl) {
        console.log('your browser/OS/drivers do not support WebGL2');
        console.log('Switching to WebGL1');
        const gl = canvas.getContext("webgl2");
        const ext1 = gl.getExtension('OES_texture_float');
        if (!ext1) {
            console.log("Sorry, your browser does not support OES_texture_float. Use a different browser");
            // return;
        }

    } else {
        console.log('webgl2 works!');
        const ext2 = gl.getExtension('EXT_color_buffer_float');
        if (!ext2) {
            console.log("Sorry, your browser does not support  EXT_color_buffer_float. Use a different browser");
            // return;
        }
    }

    gl.disable(gl.DITHER);


    twgl.addExtensionsToContext(gl);


    const init_cameraTarget = [0, 0, 0];
    const init_camera_radius = 2.2;
    const init_cameraPosition = m4.scaleVector(m4.normalize([1, 1, 1]), init_camera_radius);
    const init_zNear = init_camera_radius / 100;
    const init_zFar = init_camera_radius * 3.0;
    const camera_up = [0, 1, 0];


    const spsElement = document.querySelector("#sps");
    const fpsElement = document.querySelector("#fps");

    const spsNode = document.createTextNode("");
    const fpsNode = document.createTextNode("");

    spsElement.appendChild(spsNode);
    fpsElement.appendChild(fpsNode);


    const params = {
        metadataJson: 'data/metadata.json',
        metadata: null,
        models: null,
        model_type: "large",

        brushSize: 0.12,
        autoFill: true,
        debug: false,
        our_version: true,
        zoom: 1.0,
        alignment: 0,
        rotationAngle: 0,
        bumpiness: 0,
        hardClamp: true,

        enable_normal_map: true,
        enable_ao_map: true,
        enable_roughness_map: true,
        enable_grafting: true,
        point_light_strength: 0.5,
        ambient_light_strength: 0.5,

        visMode: 0, // ["color", "albedo", "normal", "height", "roughness", "ao"]


        texture_name: "Honeycomb_002",
        motion_name: "0",
        target_prompt: "water_3",


        texture_img: null,
        motion_img: null,
        video_gif: null,

        texture_idx: 0,
        graft_texture_idx: 5,


        object_info: {
            name: null,
            thumbnail: null,
            scale: 1.0,
            idx: 0,
            mesh: null,
            subdivision_lvl: 2,
            actual_subdivision_lvl: 2,
            center: true,

            point_light: 1.0,
            ambient_light: 1.0
        },


        camera_locked: true,
        cameraTarget: [0, 0, 0],

        camera_radius: init_camera_radius,
        cameraPosition: init_cameraPosition,

        zNear: init_zNear,
        zFar: init_zFar,

    };


    let metadata = null;


    let gui = null;
    let currentTexture = null;
    let currentObject = null;

    const initTexture = "Skin_Lizard_002";
    const initPrompt = "cactus";
    const initObject = "sphere"

    let last_cursor_style = "default";

    initMetaData();

    async function initMetaData(load_meta_data = true) {
        if (load_meta_data) {
            const r = await fetch(params.metadataJson);
            metadata = await r.json();
            params.metadata = metadata;
            initUI();
        } else {
            metadata = params.metadata
        }


        let texture_names_ordered = metadata['texture_names_ordered']
        let texture_names = metadata['texture_names'];
        // let texture_images = metadata['texture_images'];

        let motion_names = metadata['motion_names'];
        let motion_images = metadata['motion_images'];

        // let object_names = metadata['object_names'];
        // let object_images = metadata['object_images'];

        let object_info_set = metadata['object_info'];
        let object_names = Object.keys(object_info_set);


        // let vec_field_model_files = metadata['vec_field_model_files'];
        // let video_model_files = metadata['video_model_files'];

        let target_prompts = metadata['target_prompts'];
        // let video_appearance_images = metadata['video_appearance'];
        // let video_gifs = metadata['video_gifs'];

        async function setTextureModel(idx) {

            params.texture_name = texture_names[idx];
            params.texture_img = "images/target_textures/" + texture_names[idx] + "/albedo.png"

            params.modelSet = "data/models.json";

            // if (params.model_type === "large") {
            //     params.modelSet = ("data/image_target/large/16_128_Image_new_"
            //         + texture_names[idx]
            //         + "_sphere_6_base_None_ext_height_normal_roughness_ambient.json");
            //     params.modelSet = "data/Image_new.json";
            // } else {
            //     params.modelSet = ("data/image_target/small/12_96_Image_new_"
            //         + texture_names[idx]
            //         + "_sphere_6_base_None_ext_height_normal_roughness_ambient.json");
            // }

            params.texture_idx = texture_names_ordered.indexOf(params.texture_name);
            updateUI();
            await reload_nca_weights();
            await updateCA();


        }

        let len = texture_names.length;
        for (let idx = 0; idx < len; idx++) {
            let media_path = "";
            let texture_name = "";

            const texture = document.createElement('div');


            texture_name = texture_names[idx];
            media_path = "images/target_textures/" + texture_names[idx] + "/albedo.png"
            texture.style.background = "url('" + media_path + "')";
            texture.style.backgroundSize = "100%100%";
            texture.className = 'texture-square';


            // texture.style.backgroundSize = "100px100px";
            texture.id = name; //html5 support arbitrary id:s

            texture.onclick = () => {
                // removeOverlayIcon();
                currentTexture.style.borderColor = "white";
                currentTexture = texture;
                texture.style.borderColor = "rgb(245 140 44)";
                if (!window.matchMedia('(min-width: 500px)').matches && navigator.userAgent.includes("Chrome")) {
                    texture.scrollIntoView({behavior: "smooth", block: "nearest", inline: "center"})
                }
                setTextureModel(idx);
            };
            let gridBox = $('#texture');

            if (texture_name === initTexture) {
                currentTexture = texture;
                texture.style.borderColor = "rgb(245 140 44)";
                gridBox.prepend(texture);
                setTextureModel(idx);

            } else {
                gridBox.insertBefore(texture, gridBox.lastElementChild);
            }


        }

        async function set3DObject(idx) {
            let object_name = object_names[idx];
            let object_info = object_info_set[object_name];
            params.object_info.idx = idx;
            params.object_info.name = object_name;
            params.object_info.thumbnail = object_info["thumbnail"];
            params.object_info.scale = object_info['scale'];
            params.object_info.center = object_info['center']

            // updateUI();
            // await reload_mesh();
            reset_camera();
            await reload_mesh_data();
            await updateCA();
            params.object_info.point_light = object_info['point_light'];
            params.object_info.ambient_light = object_info['ambient_light'];
            // if (ca != null) {
            //     ca.clearCircle(0, 0, 1000);
            //     ca.paint(0, 0, 10000, params.motion_idx, [0, 0]);
            // }

            // updateUI();
        }


        for (let idx = 0; idx < object_names.length; idx++) {
            let object_name = object_names[idx];
            const obj = document.createElement('div');
            obj.style.background = "url('" + object_info_set[object_name]['thumbnail'] + "')";
            obj.style.backgroundSize = "100%100%";
            obj.id = name; //html5 support arbitrary id:s
            obj.className = 'texture-square';
            obj.onclick = () => {
                // removeOverlayIcon();
                currentObject.style.borderColor = "white";
                currentObject = obj;
                currentObject.style.borderColor = "rgb(245 140 44)";
                if (!window.matchMedia('(min-width: 500px)').matches && navigator.userAgent.includes("Chrome")) {
                    currentObject.scrollIntoView({behavior: "smooth", block: "nearest", inline: "center"})
                }
                // setMotionModel(idx);
                set3DObject(idx);
            };
            let gridBox = $('#mesh');

            if (object_name == initObject) {
                currentObject = obj;
                currentObject.style.borderColor = "rgb(245 140 44)";
                gridBox.prepend(currentObject);
                set3DObject(idx);
            } else {
                gridBox.insertBefore(obj, gridBox.lastElementChild);
            }


        }


        //
        // $$(".pattern-selector").forEach(sel => {
        //     sel.onscroll = () => {
        //         alret("scroll");
        //         removeOverlayIcon();
        //         sel.onscroll = null;
        //     }
        // });


    }

    function canvasToGrid(x, y) {
        const [w, h] = ca.gridSize;
        const gridX = x / canvas.clientWidth;
        const gridY = y / canvas.clientHeight;
        return [gridX, gridY];
    }

    function getMousePos(e) {
        return canvasToGrid(e.offsetX, e.offsetY);
    }

    function createCA() {
        ca = new MeshNCA(gl, params.models, [W, H], gui, params.our_version, params.object_info.mesh, params.texture_idx, params.graft_texture_idx);

        ca.clearCircle(0, 0, 10000, null, true);
        ca.resetGraft();
        ca.alignment = params.alignment;
        ca.rotationAngle = params.rotationAngle;
        ca.hardClamp = params.hardClamp;


    }


    function getTouchPos(touch) {
        const rect = canvas.getBoundingClientRect();
        return canvasToGrid(touch.clientX - rect.left, touch.clientY - rect.top);
    }

    let prevPos = [0, 0]

    function reset_camera() {
        params.cameraPosition = init_cameraPosition;
        params.cameraTarget = init_cameraTarget;
        params.camera_radius = init_camera_radius;


    }

    function rotate_camera(delta_phi, delta_theta) {
        const up = [0, 1, 0];
        let z_camera = m4.normalize(m4.subtractVectors(params.cameraTarget, params.cameraPosition));

        let y_camera = m4.normalize(m4.subtractVectors(up, m4.scaleVector(up, m4.dot(z_camera, up))));

        let x_camera = m4.normalize(m4.cross(z_camera, y_camera));

        params.cameraPosition = m4.addVectors(params.cameraPosition, m4.scaleVector(x_camera, delta_phi));
        params.cameraPosition = m4.addVectors(params.cameraPosition, m4.scaleVector(y_camera, delta_theta));
        params.cameraPosition = m4.scaleVector(m4.normalize(params.cameraPosition), params.camera_radius);

    }

    function zoom_camera(zoom_in = false) {
        if (zoom_in) {
            params.camera_radius *= 0.9;
        } else {
            params.camera_radius /= 0.9;
        }

        params.cameraPosition = m4.scaleVector(m4.normalize(params.cameraPosition), params.camera_radius);

        params.zNear = params.camera_radius / 100;
        params.zFar = params.camera_radius * 3.0;


    }

    function click(pos, e, first_touch = false) {
        const [x, y] = pos;
        const [px, py] = prevPos;


        if (!params.camera_locked || e.shiftKey) {

            let delta_phi = 0.0;
            let delta_theta = 0.0;

            if (!first_touch) {
                delta_phi = -(x - px) * 4.0;
                delta_theta = (y - py) * 4.0;
            }
            rotate_camera(delta_phi, delta_theta);
            prevPos = pos;
        } else {

            let brushSize = params.brushSize;
            // ca.clearCircle(x, y, brushSize, null, false);
            if (params.enable_grafting) {
                ca.addGraft(x, y, brushSize);
            } else {
                ca.clearCircle(x, y, brushSize, null, false);
            }
            // ca.clearCircle(x, y, 100, null, params.zoom);
            // ca.paint(x, y, params.brushSize, params.model, [x - px, y - py]);
            prevPos = pos;
        }


    }


    function updateUI() {
        params.enable_normal_map = $('#enable_normal_map').checked;
        params.enable_ao_map = $('#enable_ao_map').checked;
        params.enable_roughness_map = $('#enable_roughness_map').checked;
        params.enable_grafting = $('#enable_grafting').checked;


        $('#play').style.display = paused ? "inline" : "none";
        $('#pause').style.display = !paused ? "inline" : "none";

        const speed = parseInt($('#speed').value);
        // $('#speedLabel').innerHTML = ['1/8 x', '1/4 x', '1/2 x', '1x', '2x', '4x', '8x'][speed];
        $('#speedLabel').innerHTML = ['1/4 x', '1/2 x', '1x', '2x', '4x'][speed];

        // const resolution_idx = parseInt($('#resolution').value);
        const bumpiness = parseInt($('#bumpiness').value) / 100.0;
        $('#bumpinessLabel').innerHTML = bumpiness.toFixed(2);
        params.bumpiness = bumpiness;


        const visMode = $('#visMode').selectedIndex;
        params.visMode = visMode;


        params.graft_texture_name = $('#graft_select').value;
        params.graft_texture_idx = metadata['texture_names_ordered'].indexOf(params.graft_texture_name);
        params.graft_texture_img = "images/target_textures/" + params.graft_texture_name + "/albedo.png"


        params.rotationAngle = parseInt($('#rotation').value);
        $('#rotationLabel').innerHTML = params.rotationAngle + " deg";


        $("#texture_img").style.background = "url('" + params.texture_img + "') center";
        $("#texture_img").style.backgroundSize = "100%100%";
        let dtd = document.createElement('p')
        dtd.innerHTML = params.texture_name
        // dtd.href = "https://www.robots.ox.ac.uk/~vgg/data/dtd/"
        $("#texture_hint").innerHTML = '';
        $("#texture_hint").appendChild(dtd);

        $("#graft_img").style.background = "url('" + params.graft_texture_img + "') center";
        $("#graft_img").style.backgroundSize = "100%100%";

        let oai = document.createElement('p')
        oai.innerHTML = params.graft_texture_name;
        // oai.href = "https://www.bukowskis.com/en/auctions/H042/96-franciska-clausen-contre-composition-composition-neoplasticiste-hommage-a-mondrian";
        $("#graft_hint").innerHTML = '';
        $("#graft_hint").appendChild(oai);


    }

    function initUI() {
        window.onkeyup = function (e) {
            canvas.style.cursor = "default";
            last_cursor_style = canvas.style.cursor;
        }
        window.onkeydown = function (e) {
            e.preventDefault();
            if (e.shiftKey) {
                if (canvas.style.cursor != "grabbing") {
                    canvas.style.cursor = "grab";
                    last_cursor_style = canvas.style.cursor;
                }

            }
        }

        $('#play-pause').onclick = () => {
            paused = !paused;
            updateUI();
        };
        $('#reset').onclick = () => {

            ca.clearCircle(0, 0, 10000, null, true);
            ca.resetGraft();

        };
        $('#benchmark').onclick = () => {
            ca.benchmark();
            elapsedTime = 0;
            first_draw = true;
        };


        $('#camera_toggle').onclick = () => {
            params.camera_locked = !params.camera_locked
            if (params.camera_locked) {
                $('#camera_toggle').src = "images/camera_locked.png";
                canvas.style.cursor = "default";
                last_cursor_style = canvas.style.cursor;
            } else {
                $('#camera_toggle').src = "images/camera_unlocked.png";
                canvas.style.cursor = "grab";
                last_cursor_style = canvas.style.cursor;
            }

        };

        $('#screenshot').onclick = () => {
            Screenshot();
        };


        $$('#alignSelect input').forEach((sel, i) => {
            sel.onchange = () => {
                params.alignment = i;
            }
        });

        $$('#brushSelect input').forEach((sel, i) => {
            sel.onchange = () => {
                if (i == 0) {
                    params.brushSize = 0.03;
                } else {
                    if (i == 1) {
                        params.brushSize = 0.06;
                    } else {
                        params.brushSize = 0.12;
                    }
                }
            }
        });
        $$('#subdivisionSelect input').forEach((sel, i) => {
            sel.onchange = () => {
                if (i == 0) {
                    // params.modelSet = params.modelSet.replace("large", "small");
                    // params.model_type = "small";
                    params.object_info.subdivision_lvl = 1;
                } else if (i === 1) {
                    params.object_info.subdivision_lvl = 2;
                } else {
                    // params.modelSet = params.modelSet.replace("small", "large");
                    // params.model_type = "large";
                    params.object_info.subdivision_lvl = 3;
                }

                // reload_mesh();
                updateCA(true);
            }
        });
        $$('#gridSelect input').forEach(sel => {
            sel.onchange = () => {
                params.hexGrid = sel.id == 'gridHex';
            }
        });
        $('#speed').onchange = updateSpeed;
        $('#speed').oninput = updateSpeed;


        $('#rotation').onchange = updateUI;
        $('#rotation').oninput = updateUI;
        $('#bumpiness').onchange = updateUI;
        $('#bumpiness').oninput = updateUI;

        $('#enable_ao_map').onchange = updateUI;
        $('#enable_normal_map').onchange = updateUI;
        $('#enable_roughness_map').onchange = updateUI;
        $('#enable_grafting').onchange = updateUI;

        $('#visMode').onchange = updateUI;
        $('#graft_select').onchange = updateUI;


        $('#zoomIn').onmousedown = () => {
            zoom_camera(true);
            updateUI();
        };


        $('#zoomOut').onmousedown = () => {
            zoom_camera(false);
            updateUI();
        };


        canvas.onmousedown = e => {
            e.preventDefault();
            // left click
            if (e.buttons == 1) {
                if (e.shiftKey || !params.camera_locked) {
                    canvas.style.cursor = "grabbing";
                } else if (params.enable_grafting) {
                    canvas.style.cursor = 'pointer';
                }

                click(getMousePos(e), e, true);
            }
        }
        canvas.onmousemove = e => {
            e.preventDefault();
            if (e.buttons == 1) {

                click(getMousePos(e), e, false);
            }
        }
        canvas.onmouseup = e => {
            e.preventDefault();
            canvas.style.cursor = last_cursor_style;
        }
        canvas.addEventListener("touchstart", e => {
            e.preventDefault();
            click(getTouchPos(e.changedTouches[0]), e, true);
        });
        canvas.addEventListener("touchmove", e => {
            e.preventDefault();
            for (const t of e.touches) {
                click(getTouchPos(t), e, false);
            }
        });

        canvas.addEventListener("wheel", e => {
            e.preventDefault();

            zoom_camera(e.deltaY < 0);
        });

        var myParent = document.body;

        // let graft_textures = metadata['texture_names_ordered'];
        let graft_textures = metadata['texture_names'];
        let graft_select_list = document.getElementById('graft_select');

        for (var i = 0; i < graft_textures.length; i++) {
            var graft_option = document.createElement("option");
            graft_option.value = graft_textures[i];
            graft_option.text = graft_textures[i];
            graft_select_list.appendChild(graft_option);
        }
        // document.getElementById('graft_select').setAttribute("value", "1");


        updateUI();
    }

    function Screenshot(name) {
        const uri = canvas.toDataURL();
        var link = document.createElement("a");
        link.download = params.object_info.name + "-" + params.texture_name;
        link.href = uri;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        // delete link;
    }

    function updateSpeed() {
        let speed = parseInt($("#speed").value);
        const current_steps = speed_sets[speed];
        const last_steps = speed_sets[lastSpeed];
        adjustedStepCount *= current_steps / last_steps;
        lastSpeed = speed;

        updateUI();
    }

    async function reload_mesh_data() {
        // const response = await fetch('../meshes/objects/'
        //     + params.object_info.name + '/' + params.object_info.name +
        //     '_remesh_lvl' + params.object_info.subdivision_lvl + '.obj');

        const response = await fetch('data/meshes/'
            + params.object_info.name + '/' + params.object_info.name +
            '_remesh_lvl' + "1" + '.obj');

        // const response = await fetch('data/objects/alien.obj');
        // const response = await fetch('data/objects/vase.obj');
        const obj_json = await response.text();
        // const obj = parseOBJ(text);
        const mesh = new Mesh(obj_json, gl, params.object_info.scale, params.object_info.subdivision_lvl - 1, params.object_info.center);
        params.object_info.mesh = mesh;
        params.object_info.actual_subdivision_lvl = params.object_info.subdivision_lvl;
    }

    async function reload_nca_weights() {
        const r = await fetch(params.modelSet);
        // const r = await fetch("data/edge_drop.json");
        // const r = await fetch("data/bubble_sphere.json");
        // const r = await fetch("data/checker_sphere.json");
        // const r = await fetch("data/alien.json");
        const models = await r.json();
        params.models = models;
    }

    async function updateCA(reload_mesh = false) {
        // Fetch models from json file
        const firstTime = ca == null;

        if (reload_mesh) {
            await reload_mesh_data();
        }

        if (params.object_info.mesh == null || params.modelSet == null) {
            return;
        }

        createCA();

        window.ca = ca;

        updateUI();
        // ca.step();

        ca.rotationAngle = params.rotationAngle;
        ca.alignment = params.alignment;
        ca.bumpiness = params.bumpiness;
        ca.hardClamp = params.hardClamp;

        ca.cameraPosition = params.cameraPosition;
        ca.cameraTarget = params.cameraTarget;
        ca.camera_radius = params.camera_radius;

        ca.zNear = params.zNear;
        ca.zFar = params.zFar;

        // To show the object at first if the program is paused.
        // twgl.bindFramebufferInfo(gl);
        // ca.draw_objects(0);
        if (firstTime) {
            requestAnimationFrame(render);
        }

    }


    let lastSpeed = parseInt($("#speed").value);

    let lastDrawTime = 0;
    let frameCount = 0;

    let fpsCount = 0;
    let stepCount = 0;
    let adjustedStepCount = 0;
    var elapsedTime = 0;

    let first_draw = true;

    function render(time) {
        const targetSPS = params.object_info.actual_subdivision_lvl <= 2 ? 4 : 2.0;
        // twgl.resizeCanvasToDisplaySize(gl.canvas);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

        if (elapsedTime / 1000 > 2.0) {
            elapsedTime = 1000 + elapsedTime - 1000 * (elapsedTime / 1000)
        }

        if (!isInViewport(canvas)) {
            requestAnimationFrame(render);
            first_draw = true;
            elapsedTime = 0;
            fpsCount = 0;
            stepCount = 0;
            return;
        }


        time = window.performance.now();


        if (first_draw) {
            first_draw = false;
            lastDrawTime = time;

            requestAnimationFrame(render);
            return;
        }


        elapsedTime += (time - lastDrawTime);
        fpsCount++;
        lastDrawTime = time;
        frameCount++;

        if (elapsedTime >= 1000) {
            elapsedTime -= 1000;
            // fpsElement.textContent = fpsCount.toFixed(1);  // update avg display
            // spsElement.textContent = stepCount.toFixed(1);  // update avg display

            fpsNode.nodeValue = fpsCount.toFixed(1);
            spsNode.nodeValue = stepCount.toFixed(1);

            fpsCount = 0;
            stepCount = 0;
            adjustedStepCount = 0;
        }

        ca.rotationAngle = params.rotationAngle;
        ca.alignment = params.alignment;
        ca.bumpiness = params.bumpiness;
        ca.hardClamp = params.hardClamp;


        ca.camera.cameraPosition = params.cameraPosition;
        ca.camera.cameraTarget = params.cameraTarget;
        ca.camera.camera_radius = params.camera_radius;

        ca.camera.zNear = params.zNear;
        ca.camera.zFar = params.zFar;
        ca.camera.view = params.camera_view;

        ca.enable_grafting = params.enable_grafting;
        ca.graft_idx = params.graft_texture_idx;

        // ca.hexGrid = params.hexGrid;

        if (!paused) {
            let speed = parseInt($("#speed").value);
            const steps = speed_sets[speed];


            const expected_steps = (elapsedTime / 1000.0) * targetSPS * steps;
            const steps_to_take = Math.ceil(expected_steps - adjustedStepCount);
            for (let n = 0; n < steps_to_take; n++) {
                ca.step();
                stepCount++;
                adjustedStepCount++;
            }


        }


        twgl.bindFramebufferInfo(gl);


        gl.clearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        const render_config = {
            enable_normal_map: params.enable_normal_map,
            enable_ao_map: params.enable_ao_map,
            enable_roughness_map: params.enable_roughness_map,

            point_light_strength: params.object_info.point_light,
            ambient_light_strength: params.object_info.ambient_light,

            visMode: params.visMode,
        }


        ca.draw_mesh(time, render_config);


        requestAnimationFrame(render);

    }
}
