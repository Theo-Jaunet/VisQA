let curData = [];

let mod = {lang: 9, vis: 5, cross: 5, head: 12};
let refKmean = {};

load_data_light().then(r => init(r));


async function load_data_light() {


    return [await d3.json('static/assets/data/data.json', d3.autoType), await d3.json('static/assets/data/k_median.json', d3.autoType)]
}


function drawModel(mod) {

    let svg = d3.select("#model")
    let mlen = mod.lang + mod.vis + (mod.cross * 2);
    let pad = 5

    let sqSize = (960 / mlen)

    let blockHeight = (210 - pad * 4) / 2

    let top_marg = 20;


    let headSize = sqSize - (mod.head * 2) / mod.head;

    let crossSt = Math.max((((sqSize + pad) * mod.lang)), (((sqSize + pad) * mod.vis)))

    crossSt += pad * 6;

    //LANG

    svg.append("rect")
        .attr("name", "lang")
        .attr("type", "0")
        .attr("nb", "0")
        .attr("x", pad)
        .attr("y", top_marg + pad)
        .attr("width", ((sqSize + pad) * mod.lang) + pad)
        .attr("height", blockHeight)
        .attr("fill", '#e1964b')
        .attr("stroke", '#555555')
        .attr("stroke-width", '1');


    svg.append("text")
        .attr("x", ((((sqSize + pad) * mod.lang) + pad) / 2) - 85)
        .attr("y", 15)
        .text("Language Self-Attention")
        .style("font-family", '"Raleway", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif')
        .style("color", "#222")
        .style("font-weight", "500")


    let x = pad
    let y = top_marg + pad * 2
    for (let i = 0; i < mod.lang; i++) {
        x += pad;
        svg.append("rect")
            .attr("name", "lang")
            .attr("type", "1")
            .attr("nb", i)
            .attr("x", x)
            .attr("y", y)
            .attr("width", sqSize)
            .attr("height", blockHeight - pad * 2)
            .attr("fill", 'steelblue')
            .attr("stroke", '#555555')
            .attr("stroke-width", '1');

        // function drawHeads(svg, nb, x, y, width, height)
        drawHeads(svg, 12, x, y, sqSize, blockHeight - (pad * 2), "lang_" + i)

        x += sqSize
    }


    // VIS
    svg.append("rect")
        .attr("name", "vis")
        .attr("type", "0")
        .attr("nb", "0")
        .attr("x", pad)
        .attr("y", top_marg + blockHeight + pad * 3)
        .attr("width", ((sqSize + pad) * mod.vis) + pad)
        .attr("height", blockHeight)
        .attr("fill", '#a5bb60')
        .attr("stroke", '#555555')
        .attr("stroke-width", '1')

    svg.append("text")
        .attr("x", ((((sqSize + pad) * mod.vis) + pad) / 2) - 70)
        .attr("y", 245)
        .text("Vision Self-Attention")
        .style("font-family", '"Raleway", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif')
        .style("color", "#222")
        .style("font-weight", "500")


    x = pad
    y = top_marg + (blockHeight + pad * 3) + pad
    for (let i = 0; i < mod.vis; i++) {
        x += pad;
        svg.append("rect")
            .attr("name", "vis")
            .attr("type", "1")
            .attr("nb", i)
            .attr("x", x)
            .attr("y", y)
            .attr("width", sqSize)
            .attr("height", blockHeight - pad * 2)
            .attr("fill", 'steelblue')
            .attr("stroke", '#555555')
            .attr("stroke-width", '1');


        drawHeads(svg, 12, x, y, sqSize, blockHeight - (pad * 2), "vis_" + i)
        x += sqSize

    }

    // cross

    for (let i = 0; i < mod.cross; i++) {

        svg.append("rect")
            .attr("name", "cross")
            .attr("type", "0")
            .attr("nb", i)
            .attr("x", crossSt + ((pad + (sqSize + pad * 2) * 2) * i) + pad)
            .attr("y", top_marg + pad)
            .attr("width", ((sqSize + pad) * 2) + pad)
            .attr("height", (blockHeight * 2) - pad * 2)
            .attr("fill", '#7964a0')
            .attr("stroke", '#555555')
            .attr("stroke-width", '1')


        x = crossSt + ((pad + (sqSize + pad * 2) * 2) * i) + (pad)
        y = top_marg + pad * 2
        let names = [["lv", "vl"], ["ll", "vv"]];

        for (let j = 0; j < 2; j++) {
            x += pad;
            svg.append("rect")
                .attr("name", names[j][0])
                .attr("type", "1")
                .attr("nb", i)
                .attr("x", x)
                .attr("y", y)
                .attr("width", sqSize * 0.9)
                .attr("height", blockHeight * 0.9 - pad * 2)
                .attr("fill", 'steelblue')
                .attr("stroke", '#555555')
                .attr("stroke-width", '1');


            svg.append("text")
                .attr("x", x + ((sqSize * 0.9) / 2) - 6)
                .attr("y", 15)
                .text(names[j][0])
                .style("font-family", '"Raleway", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif')
                .style("color", "#222")
                .style("font-weight", "500")


            drawHeads(svg, 12, x, y, sqSize, blockHeight * 0.9 - pad * 2, names[j][0] + "_" + i)

            svg.append("rect")
                .attr("name", names[j][1])
                .attr("type", "1")
                .attr("nb", i)
                .attr("x", x)
                // .attr("x", x+(i==1?pad:0))
                .attr("y", y + (blockHeight))
                .attr("width", sqSize * 0.9)
                .attr("height", blockHeight * 0.9 - pad * 2)
                .attr("fill", 'steelblue')
                .attr("stroke", '#555555')
                .attr("stroke-width", '1');


            svg.append("text")
                .attr("x", x + ((sqSize * 0.9) / 2) - 6)
                .attr("y", 225)
                .text(names[j][1])
                .style("font-family", '"Raleway", "HelveticaNeue", "Helvetica Neue", Helvetica, Arial, sans-serif')
                .style("color", "#222")
                .style("font-weight", "500")


            drawHeads(svg, 12, x, y + (blockHeight), sqSize, blockHeight * 0.9 - pad * 2, names[j][1] + "_" +
                i)

            x += sqSize + pad
        }
    }
}


function drawHeads(svg, nb, x, y, width, height, name) {

    let pad = 2;

    let headSize = ((height - pad) - ((pad * (nb / 2)) + pad)) / (nb / 2)

    // console.log(headSize);

    y += pad;
    for (let i = 0; i < nb; i++) {
        let offx = (i < 6 ? pad : headSize + pad * 4);
        let col = "#f3cfdb"
        if (refKmean[name + "_" + i] < 20) {
            col = "#bbd8e6" // blue
        } else if (refKmean[name + "_" + i] < 35) {
            col = "#cbecaf"
        } else if (refKmean[name + "_" + i] < 70) {
            col = "#f8e4c5"
        }

        svg.append("rect")
            .attr("id", name + "_" + i)
            .attr("x", x + pad + offx)
            .attr("y", y + (headSize + pad) * (i % (nb / 2)))
            .attr("width", headSize)
            .attr("height", headSize)
            .attr("fill", col)
            .attr("stroke", "#555555")
            .attr("stroke-width", '1');

    }
}


function init(dat) {


    let data = dat[0]
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

    console.log(data);
    curData = data;
    refKmean = dat[1]
    // data = JSON.parse(data)

    drawModel(mod);

    plotter_init(data);
    fillSelect(data.map(d => d.global_group), "#ggroup")
    fillSelect2(data.map(d => d.functions), "#function")


    //do STUFF
    //         }
    //     }
    // });
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
    console.log(bBox.width);

    let xscale = d3.scaleLinear().domain(xrange).range([0, 960]);
    let yscale = d3.scaleLinear().domain(yrange).range([0, 642]);


    svg.selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
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


