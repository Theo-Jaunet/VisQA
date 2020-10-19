let curData = [];
let imgs = [];
let baseUrl = "static/assets/images/try/"

let mod = {lang: 9, vis: 5, cross: 5, head: 12};

let refKmean = {};

let xscale;
let yscale;

let currKmean = {};

let currHeatmaps = {};

let currHeatLabels = {};

let mono_col = d3.scaleLinear().domain([0, 0.35, 1]).range(['#ffffe1', '#FEEAA9', '#cf582f']).interpolate(d3.interpolateHcl);


let asked = false;

load_data_light().then(r => init(r));


async function load_data_light() {


    return [await d3.json('static/assets/data/data.json', d3.autoType), await d3.json('static/assets/data/k_median.json', d3.autoType), await d3.json('/firstProj', d3.autoType)]
}


d3.json('static/assets/data/images.json', d3.autoType).then(d => {

    imgs = d["default"];


    let slide = $("#imSlide");

    slide.attr("max", imgs.length - 1);


    let im = new Image();


    im.onload = function () {

        let can = document.getElementById("inVis")

        let cont = can.getContext('2d');

        let rate = fixRatio2([im.width, im.height], [300, 300])

        can.width = rate[0]
        can.height = rate[1]

        cont.drawImage(im, 0, 0, rate[0], rate[1])

    };

    im.src = baseUrl + imgs[0] + ".jpg"

    // d3.select("#inVis").append("svg:image")
    //     .attr('x', 0)
    //     .attr('y', 0)
    //     .attr('width', 400)
    //     .attr('height', 350)
    //     .attr("preserveAspectRatio", 'xMidYMid meet')
    //     .attr("xlink:href", )


    return d
})


function fixRatio2(im, sv) {

    //size based
    let aspr = im[0] / im[1];
    let svAspr = sv[0] / sv[1];

    if (im[0] < sv[0] && im[1] < sv[1]) {
        // Image plus petite
        return [im[0], im[1], aspr];
    }

    if (im[0] > sv[0] || im[1] > sv[1]) {
        // Image plus grande
        let vr = sv[1] / im[1];
        let hr = sv[0] / im[0];

        if (vr < hr) {
            // Image Horizontale
            return [(sv[1] * im[0]) / im[1], sv[1]];
        } else if (vr > hr) {
            // Image Verticale
            return [sv[0], (sv[0] * im[1]) / im[0]];
        } else {
            return [sv[0], (sv[0] * im[1]) / im[0]];
        }
    }
}


function drawModel(mod) {

    let svg = d3.select("#model")
    let mlen = mod.lang + mod.vis + (mod.cross * 2);
    let pad = 5

    let xinter = 5; // RECT (STEELBLUE)
    let yinter = 5;


    let top_marg = 15; // BLOCK


    let blockHeight = ((300 - top_marg * 3) - ((yinter * 2) + top_marg)) / 2


    let block_xinter = 10;

    let rectHeight = blockHeight - yinter * 2
    let rectWidth = ((960) / mlen)

    let sqSize = rectWidth


    let crossSt = Math.max((((rectWidth + pad) * mod.lang)), (((rectWidth + pad) * mod.vis)))


    crossSt += pad * 6;

    //LANG

    svg.append("rect")
        .attr("name", "lang")
        .attr("type", "0")
        .attr("nb", "0")
        .attr("x", block_xinter)
        .attr("y", top_marg)
        .attr("width", (xinter + (rectWidth + xinter) * mod.lang))
        .attr("height", blockHeight)
        // .attr("fill", '#e1964b')
        .attr("fill", "#fff")
        .attr("stroke", '#555555')
        .attr("stroke-width", '1');


    svg.append("text")
        .attr("x", (block_xinter + ((xinter + ((rectWidth + xinter) * mod.lang)) / 2)) - 85)
        .attr("y", top_marg - 4)
        .text("Language Self-Attention")
        .style("font-family", '"Raleway", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif')
        .style("color", "#222")
        .style("font-weight", "500")


    let x = block_xinter
    let y = top_marg + yinter
    for (let i = 0; i < mod.lang; i++) {
        x += pad;
        svg.append("rect")
            .attr("name", "lang")
            .attr("type", "1")
            .attr("nb", i)
            .attr("x", x)
            .attr("y", y)
            .attr("width", rectWidth)
            .attr("height", rectHeight)
            .attr("fill", 'steelblue')
            // .attr("fill", 'steelblue')
            .attr("stroke", '#555555')
            .attr("stroke-width", '1');

        // function drawHeads(svg, nb, x, y, width, height)
        drawHeads(svg, mod.head, x, y, rectWidth, rectHeight, "lang_" + i)

        x += sqSize
    }


    // VIS
    svg.append("rect")
        .attr("name", "vis")
        .attr("type", "0")
        .attr("nb", "0")
        .attr("x", block_xinter)
        .attr("y", top_marg + blockHeight + top_marg)
        .attr("width", (xinter + ((rectWidth + xinter) * mod.vis)))
        .attr("height", blockHeight)
        .attr("fill", "#fff")
        // .attr("fill", '#a5bb60')
        .attr("stroke", '#555555')
        .attr("stroke-width", '1')

    svg.append("text")
        .attr("x", (block_xinter + ((xinter + ((rectWidth + xinter) * mod.vis)) / 2)) - 70)
        .attr("y", (blockHeight * 2 + (top_marg * 3)))
        .text("Vision Self-Attention")
        .style("font-family", '"Raleway", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif')
        .style("color", "#222")
        .style("font-weight", "500")


    x = block_xinter
    y = top_marg + (blockHeight + top_marg) + yinter


    for (let i = 0; i < mod.vis; i++) {
        x += pad;
        svg.append("rect")
            .attr("name", "vis")
            .attr("type", "1")
            .attr("nb", i)
            .attr("x", x)
            .attr("y", y)
            .attr("width", rectWidth)
            .attr("height", rectHeight)
            .attr("fill", 'steelblue')
            .attr("stroke", '#555555')
            .attr("stroke-width", '1');


        drawHeads(svg, mod.head, x, y, rectWidth, rectHeight, "vis_" + i)
        x += rectWidth

    }

    // cross

    let crossWidth = rectWidth * 2 + xinter * 3
    let crossHeight = rectHeight * 2 + xinter * 3

    let crossySt = (300 - crossHeight) / 2

    for (let i = 0; i < mod.cross; i++) {

        svg.append("rect")
            .attr("name", "cross")
            .attr("type", "0")
            .attr("nb", i)
            .attr("x", crossSt + (crossWidth * i) + (block_xinter * i))
            .attr("y", crossySt)
            .attr("width", (crossWidth))
            .attr("height", (crossHeight))
            .attr("fill", "#fff")
            // .attr("fill", '#7964a0')
            .attr("stroke", '#555555')
            .attr("stroke-width", '1')


        // x = crossSt + ((pad + (sqSize + pad * 2) * 2) * i) + (pad)
        x = (crossSt + (crossWidth * i) + (block_xinter * i)) + xinter
        y = crossySt + yinter
        let names = [["ll", "vv"], ["vl", "lv"]];

        for (let j = 0; j < 2; j++) {

            svg.append("rect")
                .attr("name", names[j][0])
                .attr("type", "1")
                .attr("nb", i)
                .attr("x", x)
                .attr("y", y)
                .attr("width", rectWidth)
                .attr("height", rectHeight)
                .attr("fill", 'steelblue')
                .attr("stroke", '#555555')
                .attr("stroke-width", '1');


            svg.append("text")
                .attr("x", x + (rectWidth / 2) - 6)
                .attr("y", y - 10)
                .text(names[j][0])
                .style("font-family", '"Raleway", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif')
                .style("color", "#222")
                .style("font-weight", "500")


            drawHeads(svg, mod.head, x, y, rectWidth, rectHeight, names[j][0] + "_" + i)

            svg.append("rect")
                .attr("name", names[j][1])
                .attr("type", "1")
                .attr("nb", i)
                .attr("x", x)
                // .attr("x", x+(i==1?pad:0))
                .attr("y", y + (rectHeight) + yinter)
                .attr("width", rectWidth)
                .attr("height", rectHeight)
                .attr("fill", 'steelblue')
                .attr("stroke", '#555555')
                .attr("stroke-width", '1');


            svg.append("text")
                .attr("x", x + (rectWidth / 2) - 6)
                .attr("y", (y + (rectHeight * 2) + yinter * 2) + 13)
                .text(names[j][1])
                .style("font-family", '"Raleway", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif')
                .style("color", "#222")
                .style("font-weight", "500")


            drawHeads(svg, mod.head, x, y + (rectHeight) + yinter, rectWidth, rectHeight, names[j][1] + "_" +
                i)

            x += rectWidth + xinter
        }
    }
}


function drawHeads(svg, nb, x, y, width, height, name) {

    let attxInter = 4;
    let attyInter = 4;

    let attWidth = (width - (attxInter * 3)) / 2;
    let attHeight = (height - (attyInter * (nb / 2) + attyInter)) / (nb / 2)

    // let headSize = Math.min(((height - pad) - ((pad * (nb / 2)) + pad)) / (nb / 2), ((width - pad) - ((pad * (nb / 2)) + pad)) / (nb / 2))


    // console.log(headSize);

    y += attyInter;
    for (let i = 0; i < nb; i++) {
        let offx = (i < (nb / 2) ? attxInter : attWidth + attxInter * 2);
        let offy = (attHeight + attyInter) * (i % (nb / 2))


        let col = getCol(refKmean[name + "_" + i])

        svg.append("rect")
            .attr("type", "2")
            .attr("id", name + "_" + i)
            .attr("x", x + offx)
            .attr("y", y + offy)
            .attr("width", attWidth)
            .attr("height", attHeight)
            .attr("fill", col)
            .attr("stroke", "rgba(19,19,19,0.82)")
            .attr("stroke-width", '1');

    }
}


function getCol(val) {


    let col = "#ffb3ba"
    if (val < 20) {
        col = "#bae1ff" // blue
    } else if (val < 35) {
        col = "#baffc9"
    } else if (val < 70) {
        col = "#ffdfbb"
    }

    return col
}

function findElems(data, thresh, base) {


    let names = Object.keys(data)

    let res = [];

    for (let i = 0; i < names.length; i++) {

        if (data[names[i]] < thresh && data[names[i]] >= base) {
            res.push(names[i])
        }
    }

    return res

}


function init(dat) {


    let data = dat[0].map((d, i) => {
        d.k_dist = dat[2]['proj'][i];
        return d
    })

    // data.lo


    // console.log(dat[2]);
    //
    // let form = new FormData();
    // form.append("units", []);
    //
    //
    // $.ajax({
    //     type: "POST",
    //     url: "/proj",
    //     processData: false,
    //     contentType: false,
    //     data: form,
    //     success: function (d) {
    //         if (d !== "over") {
    //
    //             console.log(d);

    // console.log(data);
    curData = data;
    refKmean = dat[1]
    currKmean = refKmean;
    // data = JSON.parse(data)


    $("#counter").html("Masked Heads: " + 0 + "/" + Object.keys(refKmean).length)

    drawModel(mod);
    setDPI(document.getElementById("heatm"), 960)
    plotter_init(data);
    fillSelect(data.map(d => d.global_group), "#ggroup")
    fillSelect2(data.map(d => d.functions), "#function")


    //do STUFF
    //         }
    //     }
    // });
}


function ask(data) {
    d = JSON.parse(data);

    // $("#result").html("Answer: <br> " + d.pred + " at " + (Math.round(d.confidence * 10000) / 100) + "%")
    console.log(d);

    DrawRes(d.five)
    filler(d.alignment)
    let svg = d3.select("#proj");

    // console.log("-----------------")
    // console.log(d.coords);
    // console.log(xscale(d.coords[0]));

    svg.select("#askDot").remove()

    svg.append("circle")
        .attr("cx", xscale(d.coords[0][0]))
        .attr("cy", yscale(d.coords[0][1]))
        .attr("r", "10")
        .attr("id", "askDot")
        .attr("fill", "steelblue")
        .attr("stroke", "#555555")
        .attr("stroke-width", "3")

    fillHeads(d.k_dist)
    currKmean = d.k_dist
    $(".kmeanSelected").toggleClass("kmeanSelected")
    UpdateCounter()
    asked = true;
    currHeatmaps = d.heatmaps
    currHeatLabels = d.labels
    //
}

function DrawRes(data) {

    let svg = d3.select("#res")

    svg.selectAll("*").remove()

    let barHeight = 15
    let barPad = 17

    let textPad = 10


    const sortable = Object.fromEntries(
        Object.entries(data).sort(([, a], [, b]) => a - b)
    );

    let ordered = Object.entries(data)

    let lscale = d3.scaleLinear().domain([0, 1]).range([3, 40]);

    for (let i = 0; i < ordered.length; i++) {


        svg.append("rect")
            .attr("x", 2)
            .attr("y", 5 + (((barHeight + barPad) * i)))
            .attr("height", barHeight)
            .attr("width", lscale(1))
            .attr("fill", "#f3f3f3")
            .attr("stroke", "#a9a9a9")
            .attr("strokeWidth", "1px")


        svg.append("rect")
            .attr("x", 2)
            .attr("y", 5 + (barHeight + barPad) * i)
            .attr("height", barHeight)
            .attr("width", lscale(ordered[i][1]))
            .attr("fill", (i === 0 ? "#a92234" : "steelblue"))
        // .attr("stroke", "#f3f3f3")
        // .attr("strokeWidth", "1px")


        svg.append("text")
            .attr("x", lscale(1) + textPad)
            .attr("y", 5 + ((barHeight + barPad) * i) + (barHeight / 2) + 3)
            .text(ordered[i][0])


    }
    // console.log(sortable);

}


function fillHeads(data) {

    let names = Object.keys(data);

    let svg = d3.select("#model")
    console.time("Coloring Squares")
    for (let i = 0; i < names.length; i++) {


        // console.log("Making name _: #" + names[i] + " Of value: " + data[names[i]]);
        let col = getCol(data[names[i]])


        svg.select("#" + names[i]).attr("fill", col);
    }
    console.timeEnd("Coloring Squares")
}


function fillSelect(data, id) {

    let map = new Map();
    let sel = $(id);
    // let select = document.getElementById('ggroup');

    // console.log(sel);

    for (let i = 0; i < data.length; i++) {
        map.set(data[i], "bob")
    }


    let temp = Array.from(map.keys());
    // console.log(temp);

    let mes = ""
    for (let i = 0; i < temp.length; i++) {


        sel.append(new Option(temp[i], temp[i]))
        sel.append();
    }


}


function fillSelect2(data, id) {

    let map = new Map();
    let sel = $(id);
    // let select = document.getElementById('ggroup');

    // console.log(sel);

    for (let i = 0; i < data.length; i++) {
        for (let j = 0; j < data[i].length; j++) {
            let el = data[i][j];
            if (el !== "select") {

                map.set(el.split(" ")[0], "bob")
            }

        }

    }


    let temp = Array.from(map.keys());
    // console.log(temp);

    let mes = ""
    for (let i = 0; i < temp.length; i++) {


        sel.append(new Option(temp[i], temp[i]))
        sel.append();
    }


}


function plotter_init(data) {


    let svg = d3.select("#proj");

    svg.selectAll("circle").remove();


    let svg1 = document.getElementById('proj');

    let bBox = svg1.getBBox();


    let xrange = d3.extent(data.map(d => d['k_dist'][0]));
    let yrange = d3.extent(data.map(d => d['k_dist'][1]));
    // console.log(bBox.width);

    xscale = d3.scaleLinear().domain(xrange).range([0, 960]);
    yscale = d3.scaleLinear().domain(yrange).range([0, 642]);


    svg.selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
        .attr("class", "umapDot")
        .attr("cx", d => {
            return xscale(d['k_dist'][0])
        })
        .attr("cy", d => yscale(d['k_dist'][1]))
        .attr("r", 2.5)
        .attr("stroke", "#555555")
        .style("stroke-width", "0.1")
        .style("opacity", "0.8")
        .attr("fill", d => d.reasoning_label === "bias" ? "red" : "green")

}


function filler(data) {

    let col = d3.scaleOrdinal(d3.schemeCategory10);

    let dat = Object.keys(data);

    let can = document.getElementById("inVis")
    let cont = can.getContext('2d');

    // console.log(dat);
    // console.log(data);
    cont.beginPath();

    for (let i = 0; i < dat.length; i++) {
        // cont
        // console.log(data[dat[i]].xywh[0]);
        cont.save()
        cont.strokeStyle = col(i);
        cont.fillStyle = col(i);
        cont.lineWidth = "5"
        cont.strokeRect(can.width * data[dat[i]].xywh[0], can.height * data[dat[i]].xywh[1], can.width * (data[dat[i]].xywh[2] - data[dat[i]].xywh[0]), can.height * (data[dat[i]].xywh[3] - data[dat[i]].xywh[1]));

        cont.shadowColor = "#000";
        cont.shadowOffsetX = 0;
        cont.shadowOffsetY = 0;
        cont.shadowBlur = 5;

        cont.font = '24px serif';
        cont.fillText(dat[i], can.width * data[dat[i]].xywh[0], can.height * data[dat[i]].xywh[1]);

        cont.font = '24px serif';
        cont.fillText(data[dat[i]].class, can.width * data[dat[i]].xywh[2], can.height * data[dat[i]].xywh[3]);
        cont.restore()
    }
    cont.closePath();
}


function UpdateCounter() {

    let nb = attsMaps.length;
    let total = Object.keys(currKmean).length

    $("#counter").html("Masked Heads: " + nb + "/" + total)
}


function drawHeat(data) {


    // console.log("Drawing");

    let can = document.getElementById("heatm");


    let cont = can.getContext('2d');


    cont.clearRect(0, 0, 1900, 1900)


    let marg = 15;
    let pad = 5;
    let st = 100;

    let cw = (((can.width - st) - (marg * 2)) - (pad * data[0].length)) / data[0].length;
    let ch = (((can.height - st) - (marg * 2)) - (pad * data.length)) / data.length

    // console.log(cw);

    for (let i = 0; i < data.length; i++) { // Iter Horizontally

        cont.save();
        cont.font = ' 500 24px Arial';
        cont.translate(st + marg + (cw / 2) + ((cw + pad) * i), st + marg);
        cont.rotate(-Math.PI / 4);
        cont.textAlign = "left";
        cont.fillStyle = "#1e1e1e"
        cont.fillText(currHeatLabels.textual[i % currHeatLabels.textual.length], 0, 0);
        cont.restore();

        cont.save();
        cont.font = ' 400 24px Arial';
        cont.textAlign = "right";
        cont.fillStyle = "#1e1e1e"
        cont.fillText(currHeatLabels.visual[i % currHeatLabels.visual.length], 100, st + marg + (ch / 2) + ((ch + pad) * i) + pad);
        cont.restore();


        for (let j = 0; j < data[i].length; j++) { // Iter vertically
            cont.fillStyle = mono_col(data[i][j]);

            cont.fillRect(st + marg + ((cw + pad) * j), st + marg + ((ch + pad) * i) + pad, cw, ch)


        }
    }
}


function setDPI(canvas, dpi) {
    // Set up CSS size.
    canvas.style.width = canvas.style.width || canvas.width + 'px';
    canvas.style.height = canvas.style.height || canvas.height + 'px';

    // Get size information.
    var scaleFactor = dpi / 96;
    var width = parseFloat(canvas.style.width);
    var height = parseFloat(canvas.style.height);

    // Backup the canvas contents.
    var oldScale = canvas.width / width;
    var backupScale = scaleFactor / oldScale;
    var backup = canvas.cloneNode(false);
    backup.getContext('2d').drawImage(canvas, 0, 0);

    // Resize the canvas.
    var ctx = canvas.getContext('2d');
    canvas.width = Math.ceil(width * scaleFactor);
    canvas.height = Math.ceil(height * scaleFactor);

    // Redraw the canvas image and scale future draws.
    ctx.setTransform(backupScale, 0, 0, backupScale, 0, 0);
    ctx.drawImage(backup, 0, 0);
    ctx.setTransform(scaleFactor, 0, 0, scaleFactor, 0, 0);
    // let cont = can.getContext('2d');

    ctx.scale(0.1, 0.1)
}