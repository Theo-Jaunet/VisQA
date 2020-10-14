let attsMaps = [];

function block2names(name, nb) {

    let names = [];

    if (name === "lang" || name === "vis") {


        for (let i = 0; i < mod[name]; i++) {

            for (let j = 0; j < mod.head; j++) {

                names.push(name + "_" + i + "_" + j)
            }
        }
    } else {
        console.log(nb);
        for (let j = 0; j < mod.head; j++) {

            names.push("lv" + "_" + nb + "_" + j)
            names.push("ll" + "_" + nb + "_" + j)
            names.push("vl" + "_" + nb + "_" + j)
            names.push("vv" + "_" + nb + "_" + j)

        }

    }
    return names
}

function layer2names(name, nb) {

    let names = [];

    if (name === "lang" || name === "vis") {
        for (let j = 0; j < mod.head; j++) {
            names.push(name + "_" + nb + "_" + j)
        }
    } else {
        for (let j = 0; j < mod.head; j++) {
            names.push(name + "_" + nb + "_" + j)
        }
    }

    return names
}


function union_arrays(x, y) {
    let obj = {};
    for (let i = x.length - 1; i >= 0; --i)
        obj[x[i]] = x[i];
    for (let i = y.length - 1; i >= 0; --i)
        obj[y[i]] = y[i];
    let res = [];
    for (let k in obj) {
        if (obj.hasOwnProperty(k))  // <-- optional
            res.push(obj[k]);
    }
    return res;
}
