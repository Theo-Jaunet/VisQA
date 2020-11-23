let curData = [];
let imgs = [];
let baseUrl = "static/assets/images/oracle/"

let imgsBlock = {}
let mod = {lang: 9, vis: 5, cross: 5, head: 4};
// let mod = {lang: 9, vis: 5, cross: 5, head: 12};

let refKmean = {};

let kmods = 1
let hideResbool = false;

let xscale;
let yscale;

let currKmean = {};
let currHeatmaps = {};
let currHeatLabels = {};
let curImg = 0
let currDiffHeat = {};


let oldKmean = {};
let oldHeatmaps = {};
let oldHeatLabels = {};


let mono_col = d3.scaleLinear().domain([0, 0.35, 1]).range(['#ffffe1', '#FEEAA9', '#9b4a28']).interpolate(d3.interpolateHcl);
let diff_col = d3.scaleLinear().domain([0, 0.08, 0.5, 0.9]).range(["EFF0E8", '#f2e7e9', "#aa5b65", "#6b111c"]).interpolate(d3.interpolateHcl);
let fDuff_col = d3.scaleSequential(d3.interpolatePuOr).domain([-0.8, 0.8]);

let asked = false;
let diff_bool = false;
let fdiff_bool = false;

let models = {};

let metaDat = {};

let modType = "oracle";
let disp = "tiny_oracle"
let ogSize = [];

let imShown;

let curHeat;
let curname;
let order;
let initFlatDone = false;
let loadedImgs = [0, 0];
let headStat = {};

let currQuest;
load_data_light().then(r => init(r));


async function load_data_light() {


    return [
        await d3.json('static/assets/data/data.json', d3.autoType),
        await d3.json('static/assets/data/k_median.json', d3.autoType),
        await d3.json('static/assets/data/mods.json', d3.autoType),
        await d3.json('static/assets/data/info.json', d3.autoType),
        await d3.json('static/assets/data/tiny_oracle_full.json', d3.autoType),
        await d3.json('static/assets/data/lxmert_tiny_full.json', d3.autoType),
        await d3.json('static/assets/data/lxmert_tiny_init_oracle_pretrain_full.json', d3.autoType)
    ]
}

function switchMod(dat) {


    let form = new FormData();
    form.append("name", dat.data);
    form.append("mod", JSON.stringify(dat.mod));
    form.append("disp", dat.display);
    form.append("type", dat.type);
    modType = dat.data;
    asked = false;
    diff_bool = false;
    mod = dat.mod
    disp = dat.display
    UpdateCounter()
    drawModel(mod)
    d3.select("#model").transition().duration(470).style("opacity", "1");
    $("#loader").css("visibility", "hidden")
    $("#ask").click();

    // $.ajax({
    //     type: "POST",
    //     url: "/switchMod",
    //     processData: false,
    //     contentType: false,
    //     data: form,
    //     success: function (d) {
    //         baseUrl = "static/assets/images/" + (dat.data === 'default' ? 'try' : dat.data) + "/"
    //         // imgs = imgsBlock[dat.name]
    //         let slide = $("#imSlide");
    //
    //         // d3.select("#sceneGraph").selectAll("*").remove();
    //         //
    //         // slide.attr("max", imgs.length - 1);
    //         // loadImg(baseUrl + imgs[0] + ".jpg");
    //         // if (dat.name !== 'oracle') {
    //         //     $("#productName").html('')
    //         // } else {
    //         //     fillQuest(imgs[0])
    //         // }
    //         mod = dat.mod
    //         UpdateCounter()
    //         drawModel(mod)
    //         d3.select("#model").transition().duration(470).style("opacity", "1");
    //         $("#loader").css("visibility", "hidden")
    //
    //     }
    // })


}


d3.json('static/assets/data/images.json', d3.autoType).then(d => {

    imgs = d["oracle"];

    imgsBlock = d
    let slide = $("#imSlide");

    slide.attr("max", imgs.length - 1);

    loadImg(baseUrl + imgs[0] + ".jpg")


    return d
})


function loadImg(src) {

    let im = new Image();


    im.onload = function () {

        let can = document.getElementById("inVis")

        let cont = can.getContext('2d');

        let rate = fixRatio2([im.width, im.height], [350, 300])

        can.width = rate[0]
        can.height = rate[1]

        cont.drawImage(im, 0, 0, rate[0], rate[1])

        imShown = im

    };

    im.src = baseUrl + imgs[0] + ".jpg"

}


function loadImg2(src, x, y, w, h, name, wr, hr) {


    let can = document.getElementById("inVis")

    let cont = can.getContext('2d');

    let rate = fixRatio2([imShown.width, imShown.height], [350, 300])

    can.width = rate[0]
    can.height = rate[1]
    cont.strokeStyle = "red";
    cont.drawImage(imShown, 0, 0, rate[0], rate[1])

    cont.fillStyle = "red";
    cont.lineWidth = "2"
    cont.strokeRect(x * wr, y * hr, w * wr, h * hr)

    cont.font = '24px serif';
    let tx = 5
    let ty = 20
    cont.shadowColor = "#000";
    cont.shadowOffsetX = 0;
    cont.shadowOffsetY = 0;
    cont.shadowBlur = 1;

    if (x * wr < 60 && y * hr < 40) {
        ty = can.height - 20
    }
    cont.fillText(name, tx, ty);


}


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

    svg.selectAll("*").remove();
    let mlen = mod.lang + mod.vis + (mod.cross * 2);
    let pad = 5

    let xinter = 5; // RECT (STEELBLUE)
    let yinter = 5;


    let top_marg = 18; // BLOCK


    let blockHeight = ((268 - top_marg * 3) - ((yinter * 2) + top_marg)) / 2


    let block_xinter = 20;

    let rectHeight = blockHeight - yinter * 2
    let rectWidth = ((980) / mlen)

    let sqSize = rectWidth


    let crossSt = Math.max((((rectWidth + pad) * mod.lang)), (((rectWidth + pad) * mod.vis)))

    let crossWidth = rectWidth * 2 + xinter * 3
    let crossHeight = rectHeight * 2 + xinter * 3

    let crossySt = (260 - crossHeight) / 2
    crossSt += pad * 11;

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

    svg.append("path")
        .attr("d", `M${((xinter + (rectWidth + xinter) * mod.lang) + block_xinter)} ${(top_marg + blockHeight / 2)}    C${((xinter + (rectWidth + xinter) * mod.lang) + block_xinter) + 20}  ${(top_marg + blockHeight / 2) - 4}      ${crossSt - 20} ${(crossySt + crossHeight / 2)}    ${crossSt} ${crossySt + crossHeight / 2 - 4}`)
        .attr("stroke", "rgba(115,115,115,0.53)")
        .attr("fill", "none")
        .attr("stroke-width", "2")
        .attr("stroke-dasharray", "9,2")


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
            .attr("class", "subBlock")
            .attr("name", "lang")
            .attr("type", "1")
            .attr("nb", i)
            .attr("x", x)
            .attr("y", y)
            .attr("width", rectWidth)
            .attr("height", rectHeight)
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


    let rect_twidth = (xinter + ((rectWidth + xinter) * mod.vis)) + pad * 4
    let rect_th = top_marg + blockHeight + top_marg + blockHeight / 2

    svg.append("path")
        .attr("d", `M${rect_twidth} ${rect_th}    C${rect_twidth + 100}  ${rect_th + 10}      ${crossSt - 100} ${(crossySt + crossHeight / 2) - 10}    ${crossSt} ${crossySt + crossHeight / 2}`)
        .attr("stroke", "rgba(115,115,115,0.53)")
        .attr("fill", "none")
        .attr("stroke-width", "2")
        .attr("stroke-dasharray", "9,2")


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
            .attr("class", "subBlock")
            .attr("name", "vis")
            .attr("type", "1")
            .attr("nb", i)
            .attr("x", x)
            .attr("y", y)
            .attr("width", rectWidth)
            .attr("height", rectHeight)
            .attr("stroke", '#555555')
            .attr("stroke-width", '1');


        drawHeads(svg, mod.head, x, y, rectWidth, rectHeight, "vis_" + i)
        x += rectWidth

    }

    // cross


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


        let tw = crossSt + (crossWidth * (i + 1)) + (block_xinter * (i))
        let th = crossySt + crossHeight / 2

        console.log("LLAAAA --" + tw + "--" + th + " ||" + (tw + block_xinter));
        svg.append("path")
            .attr("d", `M${tw} ${th} ${tw + block_xinter} ${th}`)
            .attr("stroke", "rgba(115,115,115,0.53)")
            .attr("fill", "none")
            .attr("stroke-width", "2")
            .attr("stroke-dasharray", "9,2")


        // x = crossSt + ((pad + (sqSize + pad * 2) * 2) * i) + (pad)
        x = (crossSt + (crossWidth * i) + (block_xinter * i)) + xinter
        y = crossySt + yinter
        let names = [["vl", "lv"], ["ll", "vv"]];

        for (let j = 0; j < 2; j++) {

            svg.append("rect")
                .attr("name", names[j][0])
                .attr("type", "1")
                .attr("nb", i)
                .attr("x", x)
                .attr("y", y)
                .attr("width", rectWidth)
                .attr("height", rectHeight)
                .attr("class", "subBlock")
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
                .attr("class", "subBlock")
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
            .attr("class", "rectMod")
            .attr("type", "2")
            .attr("id", name + "_" + i)
            .attr("x", x + offx)
            .attr("y", y + offy)
            .attr("width", attWidth)
            .attr("height", attHeight)
            .attr("ow", attWidth)
            .attr("oh", attHeight)
            .attr("fill", col)
            .attr("stroke", "rgba(19,19,19,0.82)")
            .attr("stroke-width", '1');
    }
}


function getCol(val) {

    let col = "#ffb3ba"
    if (val < 12) {
        col = "#bae1ff" // blue
    } else if (val < 25) {
        col = "#baffc9"
    } else if (val < 50) {
        col = "#ffdfbb"
    }

    return col
}

function findElems(data, thresh, base) {


    let names = Object.keys(data)

    let res = [];

    for (let i = 0; i < names.length; i++) {

        if (data[names[i]][kmods] < thresh && data[names[i]][kmods] >= base) {
            res.push(names[i])
        }
    }
    return res
}


function initStacked() {
    let jqSv = $("#stacked")

    let size = [jqSv.width(), jqSv.height()]

    let leftmarg = 120

    let lineMarg = 5


    let svg = d3.select("#stacked")


    // svg.append("line")
    //     .attr("x1", leftmarg + lineMarg)
    //     .attr("y1", size[1] / 2)
    //     .attr("x2", size[0] - lineMarg)
    //     .attr("y2", size[1] / 2)
    //     .attr("stroke", "#555555")
    //     .attr("stroke-width", 1)


    svg.append("line")
        .attr("x1", leftmarg)
        .attr("y1", lineMarg)
        .attr("x2", leftmarg)
        .attr("y2", size[1] - lineMarg)
        .attr("stroke", "#555555")
        .attr("stroke-width", 1)

    let funcs = Object.keys(headStat["tiny_oracle"]["lang_0_0"]["functions"]);

    // let groups = Object.keys(headStat["lang_0_0"]["groups"]);
    let grs = Object.keys(headStat["tiny_oracle"]["lang_0_0"]["groups"]);


    let temp = funcs.map(d => [0, 0, 0, 0])
    let temp2 = grs.map(d => [0, 0, 0, 0])

    let stack = d3.stack()
        .keys([0, 1, 2, 3])


    let topStack = svg.append("g")
        .attr("id", "topStacked")

    let botStack = svg.append("g")
        .attr("id", "botStack")


    let color = ["#bae1ff", "#baffc9", "#ffdfbb", "#ffb3ba"]

    const groups = topStack.append('g')
        .attr("id", "topBars")
        .selectAll('g')
        .data(stack(temp))
        .join('g')
        .style('fill', (d, i) => color[d.key])


    const groups2 = botStack.append('g')
        .attr("id", "botBars")
        .selectAll('g')
        .data(stack(temp2))
        .join('g')
        .style('fill', (d, i) => color[d.key])


    let bandWidth = 15
    let bandPad = 6

    let yScale = d3.scaleLinear().domain([0, 300]).range([(size[1] - 120) / 2, 50])

    let yScale2 = d3.scaleLinear().domain([0, 300]).range([(size[1] - 120), size[1] / 2]);


    groups.selectAll('rect')
        .data(d => d)
        .join('rect')
        .attr('x', (d, i) => 20 + leftmarg + (bandWidth * i) + (bandPad * i))
        .attr('y', d => yScale(d[1]))
        .attr('height', d => yScale(d[0]) - yScale(d[1]))
        .attr('width', bandWidth)
        .attr("stroke", "rgba(100,100,100,0.44)")
        .attr("stroke-width", "1")


    groups2.selectAll('rect')
        .data(d => d)
        .join('rect')
        .attr('x', (d, i) => 20 + leftmarg + (bandWidth * i) + (bandPad * i))
        .attr('y', d => yScale2(d[1]))
        .attr('height', d => yScale2(d[0]) - yScale2(d[1]))
        .attr('width', bandWidth)
        .attr("stroke", "rgba(100,100,100,0.44)")
        .attr("stroke-width", "1")

    let tg = topStack.append("g")
        .attr("id", "topLabels")


    let tg2 = botStack.append("g")
        .attr("id", "botLabels")


    // for (let i = 0; i < funcs.length; i++) {
    //
    //     let tx = 20 + leftmarg + (bandWidth * i) + (bandPad * i) + bandWidth / 2
    //     let ty = (190)
    //     tg.append('text')
    //         .attr("text-anchor", "end")
    //         .style("transform", "translate(" + tx + "px," + ty + "px) rotate(-85deg)")
    //         .text(funcs[i])
    // }
    //
    //
    // for (let i = 0; i < grs.length; i++) {
    //
    //     let tx = 20 + leftmarg + (bandWidth * i) + (bandPad * i) + bandWidth / 2
    //     let ty = (size[1] - 100)
    //     tg2.append('text')
    //         .attr("text-anchor", "end")
    //         .style("transform", "translate(" + tx + "px," + ty + "px) rotate(-85deg)")
    //         .text(grs[i])
    // }
    //

    let tScale = d3.scaleLinear()
        .domain([0, 100]).nice()
        .range([0, 471]).clamp(true)

    let vals = [12, 25, 50]

    svg.append("text")
        .attr("x", leftmarg - 35)
        .attr("y", 12)
        .style("font-size", "12pt")
        .style("font-family", "arial")
        .style("font-weight", 300)
        .text("k=0")

    svg.append("text")
        .attr("x", leftmarg - 55)
        .attr("y", 485)
        .style("font-family", "arial")
        .style("font-size", "12pt")
        .style("font-weight", 300)
        .text("k=100")


    for (let i = 0; i < vals.length; i++) {
        console.log(tScale(vals[i]));

        svg.append("line")
            .attr("x1", leftmarg)
            .attr("x2", lineMarg)
            .attr("y1", tScale(vals[i]))
            .attr("y2", tScale(vals[i]))
            .attr("stroke", "rgba(85,85,85,0.67)")
            .attr("stroke-width", 1)

        svg.append("text")
            .attr("x", 2)
            .attr("y", tScale(vals[i]) - 2)
            .style("font-size", "12pt")
            .style("font-family", "arial")
            .style("font-weight", 300)
            .text(vals[i])
    }


    svg.append("path")
        .attr("d", "")
        .attr("id", "karea");


    let gcur = svg.append('g').attr("id", "cursorKmed").style('opacity', 0)

    gcur.append("line")
        .attr("x1", leftmarg)
        .attr("x2", lineMarg)
        .attr("y1", 0)
        .attr("y2", 0)
        .attr("stroke", "#882131")
        .attr("stroke-width", "2")

    // gcur.append("text")
    //     .attr("x", 2)
    //     .attr("y", -2)
    //     .style("font-size", "12pt")
    //     .style("font-family", "arial")
    //     .style("font-weight", 300)
    //     .attr("color", "#882131")
    //     .text("curr")

}


function sortQues(a, b) {

    if (a["val"][0] > b["val"][0]) {
        return -1
    } else if (b["val"][0] > a["val"][0]) {
        return 1
    } else if (a["val"][0] == b["val"][0]) {
        if (a["val"][1] > b["val"][1]) {
            return -1
        } else if (b["val"][1] > a["val"][1]) {
            return 1
        } else if (a["val"][1] == b["val"][1]) {
            if (a["val"][2] > b["val"][2]) {
                return -1
            } else if (b["val"][2] > a["val"][2]) {
                return 1
            } else if (a["val"][2] == b["val"][2]) {
                if (a["val"][3] > b["val"][3]) {
                    return -1
                } else if (b["val"][3] > a["val"][3]) {
                    return 1
                }
            }
        }

    }
    return 0
}

function updateStats(data, id) {

    let stack = d3.stack()
        .keys([0, 1, 2, 3]);

    let leftmarg = 120;
    let lineMarg = 5;
    let bandWidth = 15;
    let bandPad = 6;
    let gr = ""
    let op = ""

    if (currQuest != -1) {
        gr = currQuest["groups"]["global"]
        op = currQuest["operations"]
    }


    let g = d3.select("#topBars");
    let teKey = Object.keys(data['functions']);

    let dat = teKey.map(d => {
        return {"val": data["functions"][d], "key": d}
    });
    // let temp = dat.sort((a, b) => (a["val"][0] > b["val"][0]) ? -1 : ((b["val"][0] > a["val"][0]) ? 1 : 0))
    let temp = dat.sort((a, b) => sortQues(a, b));


    let labels = temp.map(d => d["key"]);

    // let labels = ;
    let grs = g.selectAll("g").data(stack(temp.map(d => d["val"])))
    // let grs = g.selectAll("g").data(stack(teKey.map(d => data["functions"][d])))

    let yScale = d3.scaleLinear().domain([150, 1500]).range([(486 - 265) / 2, 20]).nice()

    grs.selectAll('rect')
        .data(d => d)
        .join('rect').transition().duration(200).attr('y', d => yScale(d[1]))
        .attr('height', d => yScale(d[0]) - yScale(d[1]))

    let tg = d3.select("#topLabels")

    tg.selectAll("text").remove()


    for (let i = 0; i < labels.length; i++) {

        let tx = 20 + leftmarg + (bandWidth * i) + (bandPad * i) + bandWidth / 2
        let ty = (122)


        tg.append('text')
            .attr("text-anchor", "end")
            .style("transform", "translate(" + tx + "px," + ty + "px) rotate(-85deg)")
            .style("fill", (op.includes(labels[i]) ? "#df0106" : ''))
            .style("font-weight", (op.includes(labels[i]) ? "700" : '400'))
            .text(labels[i])
    }


    let g2 = d3.select("#botBars")
    let teKey2 = Object.keys(data['groups'])

    let dat2 = teKey2.map(d => {
        return {"val": data["groups"][d], "key": d}
    })
    let temp2 = dat2.sort((a, b) => sortQues(a, b))


    let labels2 = temp2.map(d => d["key"]);

    // let labels = ;

    let grs2 = g2.selectAll("g").data(stack(temp2.map(d => d["val"])))
    // let grs = g.selectAll("g").data(stack(teKey.map(d => data["functions"][d])))

    let yScale2 = d3.scaleLinear().domain([50, 500]).range([486 - 130, 486 - 220])

    grs2.selectAll('rect')
        .data(d => d)
        .join('rect').transition().duration(200).attr('y', d => yScale2(d[1]))
        .attr('height', d => yScale2(d[0]) - yScale2(d[1]))

    let tg2 = d3.select("#botLabels")

    tg2.selectAll("text").remove()

    for (let i = 0; i < labels2.length; i++) {

        let tx = 20 + leftmarg + (bandWidth * i) + (bandPad * i) + bandWidth / 2
        let ty = (486 - 90)
        tg2.append('text')
            .attr("text-anchor", "end")
            .style("transform", "translate(" + tx + "px," + ty + "px) rotate(-85deg)")
            .style("fill", (gr == labels2[i]) ? "#df0106" : '')
            .style("font-weight", (gr == labels2[i]) ? "700" : '400')
            .text(labels2[i])
    }


    drawKaera(data["kmeds"], id)

}

function drawKaera(data, id) {
    var counts = {};

    let path = d3.select("#karea")

    let yScale = d3.scaleLinear()
        .domain([0, 100]).nice()
        .range([0, 471]).clamp(true)


    let xScale = d3.scaleLinear()
        .domain([0, 200]).nice()
        .range([120, 10]).clamp(true)


    for (var i = 0; i < data.length; i++) {
        var num = data[i];
        counts[num] = counts[num] ? counts[num] + 1 : 1;
    }

    let tdat = Object.keys(counts).map(d => {
        return {"key": d, "val": counts[d]}
    });


    d3.select("#cursorKmed").style("opacity", 1)
        .transition().duration(300).attr("transform", "translate(" + 0 + "," + yScale(currKmean[id][kmods]) + ")")

    // console.log(tdat);

    let area = d3.area()
        .curve(d3.curveBasis)
        .x1(d => xScale(d.val))
        .x0(xScale(0))
        .y(d => yScale(d.key))


    path.datum(tdat).transition().duration(300)
        .attr("fill", getCol(median(data)))
        .attr("d", area)
        .attr("stroke", "rgba(100,100,100,0.44)")
        .attr("stroke-width", "1")

}


function median(values) {
    if (values.length === 0) return 0;

    values.sort(function (a, b) {
        return a - b;
    });

    var half = Math.floor(values.length / 2);

    if (values.length % 2)
        return values[half];

    return (values[half - 1] + values[half]) / 2.0;
}


function init(dat) {

    models = dat[2];
    console.log(models);
    metaDat = dat[3]
    console.log(metaDat);

    headStat["tiny_oracle"] = dat[4];
    headStat["lxmert_tiny"] = dat[5];
    headStat["lxmert_tiny_init_oracle_pretrain"] = dat[6];

    let sel = $("#models");

    for (let i = 0; i < models.length; i++) {
        sel.append(new Option(models[i].display, models[i].display))
    }

    let tres = countDistrib()
    let tscale = d3.scaleLinear().domain([0, (tres[0] + tres[1] + tres[2])]).range([0, $("#qDistrib").width()])


    let cols = ["green", "gray", "red"]
    let labels = ["tail", "middle", "head"]
    let h = $("#qDistrib").height() * 0.4 - 2
    let qdis = d3.select("#qDistrib")
    let x = 0;
    for (let i = 0; i < tres.length; i++) {

        qdis.append("rect")
            .attr("x", x)
            .attr("y", 0)
            .attr("width", tscale(tres[i]))
            .attr("height", h)
            .attr("fill", cols[i])
            .attr("stroke", "#555555")
            .attr("stroke-width")

        qdis.append("text")
            .attr("x", x + tscale(tres[i]) / 2)
            .attr("y", h + 14)
            .text(labels[i] + " " + tres[i])
            .style("text-anchor", "middle")


        x += tscale(tres[i])
    }

// let data = dat[0].map((d, i) => {
//     d.k_dist = dat[2]['proj'][i];
//     return d
// })

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
// curData = data;
    refKmean = dat[1]
    currKmean = refKmean;
// data = JSON.parse(data)


    order = Object.keys(metaDat).map(d => {
        return {"score": metaDat[d]["score"], "id": d}
    })

    order = order.sort((a, b) => (a.score > b.score) ? -1 : ((b.score > a.score) ? 1 : 0))

    console.log(order[0]["id"]);
    fillQuest(order[0]["id"]);

    curImg = order[0]["id"]

    $("#counter").html("Masked Heads: " + 0 + "/" + Object.keys(refKmean).length)

    drawModel(mod);
    setDPI(document.getElementById("heatm"), 960)

    loadedImgs = [40, 40];
    fillFlat(order.slice(0, 40), order.slice(-40), 740, 75)


    loadInst(order[0]["id"], false)

    initStacked()

// plotter_init(data);
// fillSelect(data.map(d => d.global_group), "#ggroup")
// fillSelect2(data.map(d => d.functions), "#function")


//do STUFF
//         }
//     }
// });
}


function linkInput(imgW, imgH) {


    let svg = d3.select("#inputLinks")

    svg.selectAll("*").remove()

    let ratio = d3.scaleLinear().domain([193, 352]).range([15, 180])


    svg.append("path")
        .attr("d", `M${205} ${20}    C${250}  ${5}      ${250} ${180}    ${305} ${155}`)
        .attr("stroke", "rgba(85,85,85,0.7)")
        .attr("fill", "none")
        .attr("stroke-width", "2")
        .attr("stroke-dasharray", "9,2")


    svg.append("path")
        .attr("d", `M${ratio(imgW)} ${55 + imgH / 2}    C${ratio(imgW) + 80}  ${35 + imgH / 2}      ${230} ${300}    ${305} ${275}`)
        .attr("stroke", "rgba(85,85,85,0.7)")
        .attr("fill", "none")
        .attr("stroke-width", "2")
        .attr("stroke-dasharray", "9,2")


}

function loadInst(imgId, thead) {


    let questId = metaDat[imgId]["ids"]["max"];
    if (thead || questId == undefined) {
        questId = (metaDat[imgId]["ids"]["min"] !== undefined ? metaDat[imgId]["ids"]["min"] : questId)
    }


    let im = new Image();
    let val = imgId

    im.onload = function () {

        let can = document.getElementById("inVis")

        let cont = can.getContext('2d');

        imShown = im
        let rate = fixRatio2([im.width, im.height], [350, 300])
        linkInput(rate[0], rate[1])

        can.width = rate[0]
        can.height = rate[1]

        cont.drawImage(im, 0, 0, rate[0], rate[1])
    };

    im.src = baseUrl + val + ".jpg"
    if (modType === 'oracle') {
        fillQuest(val)
    }

    let q = getQ(metaDat[imgId]["questions"], questId)


    // metaDat[imgId].filter(d => d["questionId"] == questId)[0]

    console.log(q);
    currQuest = q

    let form = new FormData();
    form.append("units", attsMaps);
    form.append("question", q["question"]);
    form.append("image", val + ".jpg");
    form.append("disp", disp);


    $.ajax({
        type: "POST",
        url: "/ask",
        processData: false,
        contentType: false,
        data: form,
        success: function (d) {
            ask(d)

            $("#ask-quest").attr("value", q["question"])
        }
    })
}


function getQ(data, id) {

    let keys = Object.keys(data)
    for (let i = 0; i < keys.length; i++) {
        if (data[keys[i]]["questionId"] == id)
            return data[keys[i]]
    }


    return undefined

}

function fillQuest(id) {

    let quests = Object.keys(metaDat[id]["questions"]);
    // console.log(quests);
    let elem = $("#productName");

    elem.html('');
    // console.log(metaDat[id]["scene"]);

    fillScene(metaDat[id]["scene"])

    for (let i = 0; i < quests.length; i++) {
        let temp = metaDat[id]["questions"][quests[i]]
        elem.append(new Option(temp.question, temp.question))
    }
}

function fillScene(data) {

    let heads = Object.keys(data.objects);

    ogSize = [data.width, data.height]
    let nodes = heads.map(d => {
        return {"id": d, "item": data.objects[d]}
    });

    let links = [];
    for (let i = 0; i < heads.length; i++) {
        for (let j = 0; j < data.objects[heads[i]].relations.length; j++) {
            let el = data.objects[heads[i]].relations[j]
            links.push({source: heads[i], target: el.object, name: el.name})
        }
    }

    const simulation = d3.forceSimulation(nodes)
        .force("charge", d3.forceManyBody())
        .force("center", d3.forceCenter(400 / 2, 400 / 2))
        .force("link", d3.forceLink(links).id(d => d.id))

    let svg = d3.select("#sceneGraph");

    svg.selectAll("*").remove();
    console.log(links);

    const link = svg.append("g")
        // .attr("stroke", "#999")
        .attr("stroke-opacity", 0.9)
        .attr("stroke-width", 2)
        .selectAll("line")
        .data(links)
        .join("line")
        .attr("stroke", d => {
            let col = "#999"
            if (d.name == "to the right of") {
                col = "red"
            } else if (d.name == "to the left of") {
                col = "green"
            } else if (d.name == "in front of") {
                col = "purple"
            } else if (d.name == "behind") {
                col = "#e8991d"
            }
            // console.log(col);
            return col
        })
    // .attr("stroke-width", d => Math.sqrt(d.value));

    const node = svg.append("g")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5)
        .selectAll("circle")
        .data(nodes)
        .join("circle")
        .attr("r", 7)
        .attr("fill", "steelblue")
        .on("mouseover", handleNodeOver)
        .on("mouseout", handleNodeOut)
    // .call(drag(simulation));

    simulation.on("tick", () => {
        link
            .attr("x1", d => (d.source.x > 6 ? (d.source.x > 300 ? 295 : d.source.x) : 6))
            .attr("y1", d => (d.source.y > 6 ? (d.source.y > 348 ? 348 : d.source.y) : 6))
            .attr("x2", d => (d.target.x > 6 ? (d.target.x > 300 ? 295 : d.target.x) : 6))
            .attr("y2", d => (d.target.y > 6 ? (d.target.y > 348 ? 348 : d.target.y) : 6));

        node
            .attr("cx", d => (d.x > 6 ? (d.x > 300 ? 295 : d.x) : 6))
            .attr("cy", d => (d.y > 6 ? (d.y > 348 ? 348 : d.y) : 6));
    });


    svg.append("circle")
        .attr("cx", 20)
        .attr("cy", 375)
        .attr("r", 5)
        .attr("fill", "green")

    svg.append("text")
        .attr("x", 5)
        .attr("y", 396)
        .text("Left")


    svg.append("circle")
        .attr("cx", 65)
        .attr("cy", 375)
        .attr("r", 5)
        .attr("fill", "red")

    svg.append("text")
        .attr("x", 45)
        .attr("y", 396)
        .text("Right")


    svg.append("circle")
        .attr("cx", 115)
        .attr("cy", 375)
        .attr("r", 5)
        .attr("fill", "purple")

    svg.append("text")
        .attr("x", 90)
        .attr("y", 396)
        .text("In Front")


    svg.append("circle")
        .attr("cx", 180)
        .attr("cy", 375)
        .attr("r", 5)
        .attr("fill", "#e8991d")

    svg.append("text")
        .attr("x", 155)
        .attr("y", 396)
        .text("Behind")


    svg.append("circle")
        .attr("cx", 230)
        .attr("cy", 375)
        .attr("r", 5)
        .attr("fill", "#999")

    svg.append("text")
        .attr("x", 215)
        .attr("y", 396)
        .text("Other")


    svg.append("line")
        .attr("x1", 2)
        .attr("y1", 356)
        .attr("x2", 398)
        .attr("y2", 356)
        .attr("stroke", "#555555")

}


function handleNodeOver() {

    let nd = d3.select(this);
    nd.transition().duration(50).attr("r", 10).attr("fill", "red")

    let dat = nd.datum().item;

    showItem(document.getElementById("inVis"), ogSize, dat.x, dat.y, dat.w, dat.h, dat.name)
}


function showItem(can, ref, x, y, w, h, name) {

    let cont = can.getContext("2d");
    let wr = can.width / ref[0]
    let hr = can.height / ref[1]
    cont.strokeStyle = "red";
    let val = $("#imSlide").val();
    // console.log(val);
    loadImg2(baseUrl + imgs[val] + ".jpg", x, y, w, h, name, wr, hr)

}

function handleNodeOut() {
    let nd = d3.select(this);


    let can = document.getElementById("inVis")

    let cont = can.getContext('2d');

    let rate = fixRatio2([imShown.width, imShown.height], [300, 300])

    can.width = rate[0]
    can.height = rate[1]

    cont.drawImage(imShown, 0, 0, rate[0], rate[1])

    nd.transition().duration(50).attr("r", 7).attr("fill", "rgb(124,101,148)")
}

// drag = simulation => {
//
//     function dragstarted(event) {
//         if (!event.active) simulation.alphaTarget(0.3).restart();
//         event.subject.fx = event.subject.x;
//         event.subject.fy = event.subject.y;
//     }
//
//     function dragged(event) {
//         event.subject.fx = event.x;
//         event.subject.fy = event.y;
//     }
//
//     function dragended(event) {
//         if (!event.active) simulation.alphaTarget(0);
//         event.subject.fx = null;
//         event.subject.fy = null;
//     }
//
//     return d3.drag()
//         .on("start", dragstarted)
//         .on("drag", dragged)
//         .on("end", dragended);
// }

function ask(data) {
    // d = JSON.parse(data);
    d = data

    // $("#result").html("Answer: <br> " + d.pred + " at " + (Math.round(d.confidence * 10000) / 100) + "%")
    console.log(d);

    DrawRes(d.five)
    // filler(d.alignment)
    // let svg = d3.select("#proj");

    // console.log("-----------------")
    // console.log(d.coords);
    // console.log(xscale(d.coords[0]));

    // svg.select("#askDot").remove()
    //
    // svg.append("circle")
    //     .attr("cx", xscale(d.coords[0][0]))
    //     .attr("cy", yscale(d.coords[0][1]))
    //     .attr("r", "10")
    //     .attr("id", "askDot")
    //     .attr("fill", "steelblue")
    //     .attr("stroke", "#555555")
    //     .attr("stroke-width", "3")

    fillHeads(d.k_dist)
    currKmean = d.k_dist
    $(".kmeanSelected").toggleClass("kmeanSelected");
    UpdateCounter()
    asked = true;
    diff_bool = false;
    currHeatmaps = d.heatmaps
    currHeatLabels = d.labels
    //
}

function DrawRes(data) {

    let svg = d3.select("#model")


    svg.selectAll(".res").remove()

    let barHeight = 15
    let barPad = 17

    let textPad = 10;

    let leftMarg = 1115
    let topMarg = 65
    let opa = (hideResbool ? 0 : 1);


    // const sortable = Object.fromEntries(
    //     Object.entries(data).sort(([, a], [, b]) => a - b)
    // );

    let ordered = Object.entries(data)

    let lscale = d3.scaleLinear().domain([0, 1]).range([3, 40]);

    for (let i = 0; i < ordered.length; i++) {

        svg.append("rect")
            .attr("x", leftMarg + 2)
            .attr("y", topMarg + 5 + (((barHeight + barPad) * i)))
            .attr("height", barHeight)
            .attr("width", lscale(1))
            .attr("fill", "#f3f3f3")
            .attr("stroke", "#a9a9a9")
            .attr("strokeWidth", "1px")
            .attr("class", "res")
            .style("opacity", opa)

        svg.append("rect")
            .attr("x", leftMarg + 2)
            .attr("y", topMarg + 5 + (barHeight + barPad) * i)
            .attr("height", barHeight)
            .attr("width", lscale(ordered[i][1]))
            .attr("fill", (i === 0 ? "#a92234" : "steelblue"))
            .attr("class", "res")
            .style("opacity", opa)
        // .attr("stroke", "#f3f3f3")
        // .attr("strokeWidth", "1px")

        svg.append("text")
            .attr("class", "res")
            .attr("x", leftMarg + lscale(1) + textPad)
            .attr("y", topMarg + 5 + ((barHeight + barPad) * i) + (barHeight / 2) + 3)
            .text(ordered[i][0])
            .style("opacity", opa)


        let dist = leftMarg - 1015

        svg.append("path")
            .attr("class", 'res')
            .attr("d", `M${1030} ${125 + (1 * i)} C${1015 + dist * 0.45} ${130 + (18 * i)} ${1015 + dist * 0.55} ${topMarg + 5 + (((barHeight + barPad) * i)) - 15}    ${leftMarg + 2} ${topMarg + 5 + (((barHeight + barPad) * i)) + barHeight / 2}`)
            .attr("stroke", "rgba(85,85,85,0.7)")
            .attr("fill", "none")
            .attr("stroke-width", "2")
            .attr("stroke-dasharray", "9,2")
            .style("opacity", opa)


    }
    // console.log(sortable);

}


function fillHeads(data) {

    let names = Object.keys(data);

    let svg = d3.select("#model")
    console.time("Coloring Squares")
    for (let i = 0; i < names.length; i++) {

        // console.log("Making name _: #" + names[i] + " Of value: " + data[names[i]]);
        let col = getCol(data[names[i]][kmods])


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


function drawHearLabH(cont, txt1, x1, y1) {

    cont.save();
    cont.font = ' 400 24px Arial';
    cont.textAlign = "right";
    cont.fillStyle = "#1e1e1e"
    cont.fillText(txt1, x1, y1);
    cont.restore();

}


function drawHearLabV(cont, txt1, x1, y1) {
    cont.save();
    cont.font = ' 500 24px Arial';
    cont.translate(x1, y1);
    cont.rotate(-Math.PI / 4);
    cont.textAlign = "left";
    cont.fillStyle = "#1e1e1e"
    cont.fillText(txt1, 0, 0);
    cont.restore();

}

function highlightItem(items) {

    let can = document.getElementById("inVis")
    let cont = can.getContext("2d");


    let wr = can.width / imShown.width
    let hr = can.height / imShown.height
    cont.strokeStyle = "red";

    let colors = ["red", "blue"]


    // console.log(val);


    let rate = fixRatio2([imShown.width, imShown.height], [350, 300])

    can.width = rate[0]
    can.height = rate[1]
    cont.strokeStyle = "red";
    cont.drawImage(imShown, 0, 0, rate[0], rate[1])

    for (let i = 0; i < items.length; i++) {

        // let it = ids.filter(d => {
        //     return scenes[d].name == currHeatLabels.visual[items[i]]
        // })
        //
        // if (it.length > 1) {
        //     it = scenes[it[i]]
        // } else {
        //     it = scenes[it[0]]
        // }
        let it = currHeatLabels.bboxes[items[i]]
        cont.strokeStyle = colors[i];


        cont.fillStyle = colors[i];
        cont.lineWidth = "2";

        cont.strokeRect(it[0] * wr, it[1] * hr, it[2] * wr, it[3] * hr)

        cont.font = '24px serif';
        let tx = 5
        if (i == 1) {
            tx = can.width - 20 - (currHeatLabels.visual[items[i]].length * 10)
        }
        let ty = 20
        cont.shadowColor = "#000";
        cont.shadowOffsetX = 0;
        cont.shadowOffsetY = 0;
        cont.shadowBlur = 1;

        if (it[0] * wr < 60 && it[1] * hr < 40) {
            ty = can.height - 20
        }
        cont.fillText(currHeatLabels.visual[items[i]], tx, ty);

    }


    // loadImg2(baseUrl + imgs[val] + ".jpg", x, y, w, h, name, wr, hr)

}


function coord2Inds(coords, name, data) {
    let marg = 15;
    let pad = 5;
    let st = 130;
    let can = document.getElementById("heatm");

    let cw = (((can.width - st) - (marg * 2)) - (pad * data[0].length)) / data[0].length;
    let ch = (((can.height - st) - (marg * 2)) - (pad * data.length)) / data.length

    let inds = findRC(coords, cw, ch, st, marg, pad)

    let type = name.split("_")[0]

    if (type == "vis" || type == "vv") {

        return [inds, ["v", "v"]]

    } else if (type == "lang" || type == "ll") {
        return [inds, ["l", "l"]]
    } else if (type == "vl") {
        return [inds, ["v", "l"]]
    } else if (type == "lv") {
        return [inds, ["l", "v"]]
    }

}


function getLimit(type, inds, limitV, limitL) {

    if (type == "vis" || type == "vv") {
        return inds[0] < limitV && inds[1] < limitV
    } else if (type == "lang" || type == "ll") {
        return inds[0] < limitL && inds[1] < limitL
    } else if (type == "vl") {
        return inds[0] < limitV && inds[1] < limitL
    } else if (type == "lv") {
        return inds[0] < limitL && inds[1] < limitV
    }

}


function drawHeat(data, name, coords) {

    let type = name.split("_")[0]


    let can = document.getElementById("heatm");


    let cont = can.getContext('2d');

    cont.lineWidth = "1px"
    cont.strokeStyle = "rgba(107,107,107,0.62)"

    cont.clearRect(0, 0, 1900, 1900)


    let marg = 15;
    let pad = 5;
    let st = 130;

    let cw = (((can.width - st) - (marg * 2)) - (pad * data[0].length)) / data[0].length;
    let ch = (((can.height - st) - (marg * 2)) - (pad * data.length)) / data.length
    // cont.fillRect(st + marg + ((cw + pad) * j), st + marg + ((ch + pad) * i) + pad

    // if (coords[0] > st - marg - (cw / 2) + pad && coords[1] > st - marg - (cw / 2) + pad) {

    let inds = findRC(coords, cw, ch, st, marg, pad)

    // console.log(inds);

    // if ((inds[0] >= 0 && inds[1] >= 0) && (inds[0] < currHeatLabels.visual.length && inds[1] < currHeatLabels.visual.length)) {
    if ((inds[0] >= 0 && inds[1] >= 0) && getLimit(type, inds, currHeatLabels.visual.length, currHeatLabels.textual.length)) {
        cont.fillStyle = "rgba(255,0,2,0.62)"
        cont.fillRect(0, st + marg + ((ch + pad) * inds[1]), can.width, ch + pad * 2)
        cont.fillRect(st + marg + ((cw + pad) * inds[0]) - pad, 0, cw + pad * 2, can.height)

        if (type == "vis" || type == "vv") {
            highlightItem(inds)
        } else if (type == "vl") {
            highlightItem([inds[0]])
        } else if (type == "lv") {
            highlightItem([inds[1]])
        }
    } else {
        cont.fillStyle = "rgba(21,10,255,0.62)"
        if (inds[0] < 0 && inds[1] >= 0) {

            cont.fillRect(0, st + marg + ((ch + pad) * inds[1]), can.width, ch + pad * 2)


        } else if (inds[1] < 0 && inds[0] >= 0) {
            cont.fillRect(st + marg + ((cw + pad) * inds[0]) - pad, 0, cw + pad * 2, can.height)
        }
    }

    for (let i = 0; i < data.length; i++) { // Iter Horizontally


        if (type == "lang" || type == "ll") {
            drawHearLabH(cont, currHeatLabels.textual[i % currHeatLabels.textual.length], st, st + marg + (ch / 2) + ((ch + pad) * i) + pad)
        } else if (type == "vis" || type == "vv") {
            drawHearLabH(cont, currHeatLabels.visual[i % currHeatLabels.visual.length], st, st + marg + (ch / 2) + ((ch + pad) * i) + pad)
        } else if (type == "vl") {
            drawHearLabH(cont, currHeatLabels.textual[i % currHeatLabels.textual.length], st, st + marg + (ch / 2) + ((ch + pad) * i) + pad)
        } else if (type == "lv") {
            drawHearLabH(cont, currHeatLabels.visual[i % currHeatLabels.visual.length], st, st + marg + (ch / 2) + ((ch + pad) * i) + pad)
        } else {
            drawHearLabH(cont, currHeatLabels.visual[i % currHeatLabels.visual.length], st, st + marg + (ch / 2) + ((ch + pad) * i) + pad)
        }

        for (let j = 0; j < data[i].length; j++) { // Iter vertically

            if (i == 0) {
                if (type != "vl" && type != "vis" && type != "vv") {
                    drawHearLabV(cont, currHeatLabels.textual[j % currHeatLabels.textual.length], st + marg + (cw / 2) + ((cw + pad) * j), st + marg)
                } else {
                    drawHearLabV(cont, currHeatLabels.visual[j % currHeatLabels.visual.length], st + marg + (cw / 2) + ((cw + pad) * j), st + marg)
                }
            }

            if (diff_bool) {
                cont.fillStyle = diff_col(data[i][j]);
            } else if (fdiff_bool) {
                cont.fillStyle = fDuff_col(data[i][j]);
            } else {
                cont.fillStyle = mono_col(data[i][j]);
            }

            // cont.fillRect(st + marg + ((cw + pad) * j), st + marg + ((ch + pad) * i) + pad, cw, ch)
            cont.fillRect(st + marg + ((cw + pad) * j), st + marg + ((ch + pad) * i) + pad, cw, ch)
            cont.strokeRect(st + marg + ((cw + pad) * j), st + marg + ((ch + pad) * i) + pad, cw, ch)
        }
    }
}


function agDiff(data) {
    let tres = data.map(d => Math.max(...d))
    return tres.reduce((a, b) => a + b) / tres.length

}

function findRC(coords, cw, ch, st, marg, pad) {

    let xscale = d3.scaleLinear().domain([88, 500]).range([145, 985])
    let yscale = d3.scaleLinear().domain([75, 500]).range([150, 990])
    // console.log(coords[1]);
    // console.log('FIXXX');
    // console.log(yscale(coords[1]));
    let colx = (xscale(coords[0]) - (st)) / (pad + cw);
    let coly = (yscale(coords[1]) - (st)) / (pad + ch);


    if (colx > 0) {
        colx = Math.floor(colx - 0.05)
    }
    if (coly > 0) {
        coly = Math.floor(coly - 0.05)
    }
    return [colx, coly]
}

function makeDiff(mat1, mat2) {


    return mat1.map((d, i) => d.map((f, j) => Math.abs(f - mat2[i][j])));

}


function makeDiff2(mat1, mat2) {

    return mat1.map((d, i) => d.map((f, j) => f - mat2[i][j]));
}


async function loadImgs(array, size) {

    let res = [];
    for (let i = 0; i < array.length; i++) {

        let im = new Image();


        im.onload = function () {

            let rate = fixRatio2([im.width, im.height], size)
            im.width = rate[0]
            im.height = rate[1]


        };

        im.src = baseUrl + array[i] + ".jpg"

        res.push(im)
    }


    return res
}


function addImageProcess(src, size) {
    return new Promise((resolve, reject) => {
        let img = new Image()
        img.onload = () => {
            let rate = fixRatio2([img.width, img.height], size);
            img.width = rate[0];
            img.height = rate[1];
            return resolve(img);
        }
        img.onerror = reject
        img.src = src
    })
}

async function fillFlat(tails, heads, width, height) {

    let tail = $("#tail");
    let head = $("#head");


    for (let iter = 0; iter < tails.length; iter++) {

        let div = $('<div/>')


        if (iter === 0) {
            div.attr("class", "selectedIm")
        }
        div.attr("num", iter)
        div.css("background-image", "url(" + (baseUrl + tails[iter]["id"] + ".jpg") + ")")


        tail.append(div)

        // let img = await addImageProcess(baseUrl + tails[iter]["id"] + ".jpg", [height, height])
        //
        // img.setAttribute("num", iter)
        // if (iter == 0) {
        //     img.setAttribute("class", "selectedIm")
        // }
        // tail.append(img)


    }


    for (let iter = 0; iter < heads.length; iter++) {
        // let img = await addImageProcess(baseUrl + heads[iter]["id"] + ".jpg", [height, height])
        // img.setAttribute("num", order.length - heads.length + iter)
        // head.append(img)


        let div = $('<div/>')

        div.attr("num", order.length - heads.length + iter)
        div.css("background-image", "url(" + (baseUrl + heads[iter]["id"] + ".jpg") + ")")


        head.append(div)

    }
    head.animate({scrollLeft: loadedImgs[0] * 60}, 0);

    initFlatDone = true

    // fixRatio2()

}


function getHighHeads(ids, types) {

    let thresh = 0.5;

    let keys = Object.keys(currHeatmaps)

    let heads = {};

    for (let i = 0; i < keys.length; i++) {

        let heat = currHeatmaps[keys[i]]

        if (ids.length === 1) {

            if (types[0] === "l") {

                if (keys[i].includes("lang") || keys[i].includes("ll")) {

                    if (checkRowHeat(heat, ids[0]) > thresh) {
                        heads[keys[i]] = true
                    } else if (checkColHeat(heat, ids[0]) > thresh) {
                        heads[keys[i]] = true
                    }

                } else if (keys[i].includes("vl")) {

                    if (checkRowHeat(heat, ids[0]) > thresh) {
                        heads[keys[i]] = true
                    }

                } else if (keys[i].includes("lv")) {
                    if (checkColHeat(heat, ids[0]) > thresh) {
                        heads[keys[i]] = true
                    }
                }

            } else if (types[0] === "v") {
                if (keys[i].includes("vis") || keys[i].includes("vv")) {
                    if (checkRowHeat(heat, ids[0]) > thresh) {
                        heads[keys[i]] = true
                    } else if (checkColHeat(heat, ids[0]) > thresh) {
                        heads[keys[i]] = true
                    }

                } else if (keys[i].includes("vl")) {
                    if (checkColHeat(heat, ids[0]) > thresh) {
                        heads[keys[i]] = true
                    }
                } else if (keys[i].includes("lv")) {
                    if (checkRowHeat(heat, ids[0]) > thresh) {
                        heads[keys[i]] = true
                    }
                }

            }
        } else if (types[0] === types[1]) { // Two of the same type
            if (types[0] === "l") {
                if (keys[i].includes("lang") || keys[i].includes("ll")) {
                    let temp = ids.slice().reverse()
                    if (checkCellHeat(heat, temp) > thresh) {
                        heads[keys[i]] = true
                    }
                }

            } else if (types[0] === "v") {
                if (keys[i].includes("vis") || keys[i].includes("vv")) {
                    let temp = ids.slice().reverse()
                    if (checkCellHeat(heat, temp) > thresh) {
                        heads[keys[i]] = true
                    }
                }
            }
        } else { // TWO of different types
            if (keys[i].includes("vl")) {
                if (types[0] == "v") {
                    let temp = ids.slice().reverse()
                    if (checkCellHeat(heat, temp) > thresh) {
                        heads[keys[i]] = true
                    }

                } else {
                    if (checkCellHeat(heat, ids) > thresh) {
                        heads[keys[i]] = true
                    }
                }
            } else if (keys[i].includes("lv")) {
                if (types[0] == "l") {

                    let temp = ids.slice().reverse()
                    // console.log((checkCellHeat(heat, temp) > thresh) + " -- " + keys[i] + "---" + checkCellHeat(heat, temp)+ "---"+ thresh);
                    if (checkCellHeat(heat, temp) > thresh) {

                        // console.log("adding");
                        heads[keys[i]] = true
                    }
                } else {
                    // console.log((checkCellHeat(heat, ids) > thresh) + " -- " + keys[i] + "---" + checkCellHeat(heat, ids)+ "---"+ thresh);
                    if (checkCellHeat(heat, ids) > thresh) {
                        heads[keys[i]] = true
                    }
                }
            }
        }
    }
    return heads
}

function checkRowHeat(data, id) {
    return Math.max(...data[id])
}

function checkColHeat(data, id) {
    let temp = data.map(d => d[id]);
    return Math.max(...temp)
}

function checkCellHeat(data, ids) {
    return parseFloat(data[ids[0]][ids[1]])
}


function countDistrib() {

    let res = [0, 0, 0]
    let imgs = Object.keys(metaDat)

    for (let i = 0; i < imgs.length; i++) {

        let tkeys = Object.keys(metaDat[imgs[i]]["questions"])

        for (let j = 0; j < tkeys.length; j++) {

            let q = metaDat[imgs[i]]["questions"][j];

            if (q["ood"] === "tail") {
                res[0] += 1
            } else if (q["ood"] === "middle") {
                res[1] += 1
            } else if (q["ood"] === "head") {
                res[2] += 1
            }

        }

    }
    return res
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


function saveSvg(svgEl, name) {
    svgEl.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    var svgData = svgEl.outerHTML;
    var preface = '<?xml version="1.0" standalone="no"?>\r\n';
    var svgBlob = new Blob([preface, svgData], {type: "image/svg+xml;charset=utf-8"});
    var svgUrl = URL.createObjectURL(svgBlob);
    var downloadLink = document.createElement("a");
    downloadLink.href = svgUrl;
    downloadLink.download = name;
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
}