import java.io.*;
import java.text.DecimalFormat;

/**
 * Created by yubzhu on 2019/9/19
 */

public class Main {

    public static void main(String[] args) throws IOException {
        // extractFeature();
        visualize("train_set/train_2068601.csv");
    }

    private static void extractFeature() throws IOException {
        Writer writer = new FileWriter("misc/train.csv");
        writer.write("Cell Index,Cell X,Cell Y,Height,Azimuth,Electrical Downtilt,Mechanical Downtilt,Frequency Band,RS Power,Cell Altitude,Cell Building Height,Cell Clutter Index,X,Y,Altitude,Building Height,Clutter Index,RSRP");
        // writer.write(",log10fMHz,log10height,log10distanceKM,log10absoluteDistanceKM\n");
        writer.write("\n");
        //"bdEq0,bdGt0Ngt20,bdGt20Ngt40,bdGt40,blocked\n");
        File[] files = new File("misc/train_set").listFiles();
        if (files == null) {
            return;
        }
        DecimalFormat decimalFormat = new DecimalFormat("0.000");
        int count = 0;
        for (File file: files) {
            /*// extract all data feature (by file)
            BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
            bufferedReader.readLine();
            int regionSize = 0;
            int buildingNumber = 0;
            while (true) {
                String string = bufferedReader.readLine();
                if (string == null) {
                    break;
                }
                String[] strings = string.split(",");
                String cellIndex = strings[0];
                double CellX = Double.parseDouble(strings[1]);
                double CellY = Double.parseDouble(strings[2]);
                double height = Double.parseDouble(strings[3]);
                int azimuth = Integer.parseInt(strings[4]);
                int electricalDowntilt = Integer.parseInt(strings[5]);
                int mechanicalDowntilt = Integer.parseInt(strings[6]);
                double frequencyBand = Double.parseDouble(strings[7]);
                double RSPower = Double.parseDouble(strings[8]);
                int cellAltitude = Integer.parseInt(strings[9]);
                int cellBuildingHeight = Integer.parseInt(strings[10]);
                int cellClutterIndex = Integer.parseInt(strings[11]);
                int X = Integer.parseInt(strings[12]);
                int Y = Integer.parseInt(strings[13]);
                int altitude = Integer.parseInt(strings[14]);
                int buildingHeight = Integer.parseInt(strings[15]);
                int clutterIndex = Integer.parseInt(strings[16]);
                double RSRP = Double.parseDouble(strings[17]);
                // region size
                regionSize += 1;
                // building number
                if (buildingHeight > 0) {
                    buildingNumber += 1;
                }
            }*/
            // extract single data feature
            BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
            bufferedReader.readLine();
            while (true) {
                String string = bufferedReader.readLine();
                if (string == null) {
                    break;
                }
                String[] strings = string.split(",");
                String cellIndex = strings[0];
                double CellX = Double.parseDouble(strings[1]);
                double CellY = Double.parseDouble(strings[2]);
                double height = Double.parseDouble(strings[3]);
                int azimuth = Integer.parseInt(strings[4]);
                int electricalDowntilt = Integer.parseInt(strings[5]);
                int mechanicalDowntilt = Integer.parseInt(strings[6]);
                double frequencyBand = Double.parseDouble(strings[7]);
                double RSPower = Double.parseDouble(strings[8]);
                int cellAltitude = Integer.parseInt(strings[9]);
                int cellBuildingHeight = Integer.parseInt(strings[10]);
                int cellClutterIndex = Integer.parseInt(strings[11]);
                int X = Integer.parseInt(strings[12]);
                int Y = Integer.parseInt(strings[13]);
                int altitude = Integer.parseInt(strings[14]);
                int buildingHeight = Integer.parseInt(strings[15]);
                int clutterIndex = Integer.parseInt(strings[16]);
                double RSRP = Double.parseDouble(strings[17]);
                /*if (height == 0 || cellBuildingHeight > height) {
                    count += 1;
                    continue;
                }*/
                /* special features */
                double evilNumber = -3; // a very small number
                double log10fMHz = Math.log10(frequencyBand);
                double log10height;
                if (height == 0) {
                    log10height = evilNumber;
                } else {
                    log10height = Math.log10(height);
                }
                // distance
                double distance = Math.sqrt(Math.pow(CellX - X, 2) + Math.pow(CellY - Y, 2));
                double log10distanceKM;
                if (distance == 0) {
                    log10distanceKM = evilNumber;
                } else {
                    log10distanceKM = Math.log10(distance / 1000);
                }
                // horizontal deltaHv
                int verticalTheta = electricalDowntilt + mechanicalDowntilt;
                double horizontalTheta = 90 - Math.atan2((Y - CellY), (X - CellX)) / Math.PI * 180.0 - azimuth;
                double deltaHv = height - distance * Math.cos(Math.toRadians(horizontalTheta)) * Math.tan(Math.toRadians(verticalTheta)) + (cellAltitude - altitude);
                // absolute distance
                double relativeDistance = Math.sqrt(Math.pow(deltaHv, 2) + Math.pow(distance * Math.sin(Math.toRadians(horizontalTheta)), 2));
                double straightDistance = Math.sqrt(Math.pow(CellX - X, 2) + Math.pow(CellY - Y, 2) + Math.pow(cellAltitude - altitude, 2));
                double absoluteDistance = Math.min(relativeDistance, straightDistance);
                double log10absoluteDistanceKM;
                if (absoluteDistance == 0) {
                    log10absoluteDistanceKM = evilNumber;
                } else {
                    log10absoluteDistanceKM = Math.log10(absoluteDistance / 1000);
                }
                // cost231-hata experience function
                double cost231HataExceptAlpha = 46.3 + 33.9 * log10fMHz - 13.82 * log10height + (44.9 - 6.55 * log10height) * log10distanceKM;
                double hR = 1.5; // a classical height
                double cost231HataSuburban = cost231HataExceptAlpha - ((1.1 * log10fMHz - 0.7) * hR - (1.56 * log10fMHz - 0.8));
                double cost231HataUrban = cost231HataExceptAlpha - (3.2 * Math.pow(Math.log10(11.75 * hR), 2) - 4.97) + 3; // 3 for Cm
                // okumura-hata experience function
                double okumuraHataExceptAlpha = 69.55 + 26.16 * log10fMHz - 13.82 * log10height + (44.9 - 6.55 * log10height) * log10distanceKM;
                double okumuraHataUrban = okumuraHataExceptAlpha - (3.2 * Math.pow(Math.log10(11.75 * hR), 2) - 4.97);
                double okumuraHataSuburban = okumuraHataExceptAlpha - ((1.1 * log10fMHz - 0.7) * hR - (1.56 * log10fMHz - 0.8));
                double okumuraHataSuburb = okumuraHataExceptAlpha - (2 * Math.pow(Math.log10(frequencyBand / 28), 2) + 5.4);
                double okumuraHataOpenSence = okumuraHataExceptAlpha - (4.78 * Math.pow(log10fMHz, 2) - 18.33 * log10fMHz + 40.94);
                // building height classification
                int bdEq0 = (buildingHeight == 0) ? 1: 0;
                int bdGt0Ngt20 = (buildingHeight > 0 && buildingHeight <= 20) ? 1: 0;
                int bdGt20Ngt40 = (buildingHeight > 20 && buildingHeight <= 40) ? 1: 0;
                int bdGt40Ngt60 = (buildingHeight > 40 && buildingHeight <= 60) ? 1: 0;
                int bdGt60 = (buildingHeight > 60) ? 1: 0;
                // special deltaHv
                double spDeltaHv = deltaHv - distance * Math.sin(Math.toRadians(horizontalTheta)) * Math.tan(Math.toRadians(verticalTheta));
                // whether blocked
                int blocked;
                if (spDeltaHv <= buildingHeight) {
                    blocked = 1;
                } else {
                    blocked = 0;
                }
                String output = string + "\n";
                if (output.contains("∞")) {
                    System.out.print("Error in {" + output + "}");
                }
                writer.write(output);
            }
        }
        System.out.println("count value: " + count);
        writer.flush();
    }

    private static void visualize(String inputCsvName) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new FileReader("misc/" + inputCsvName));
        bufferedReader.readLine();

        String firstLine = bufferedReader.readLine();
        String[] firstStrings = firstLine.split(",");
        double firstCellX = Double.parseDouble(firstStrings[1]);
        double firstCellY = Double.parseDouble(firstStrings[2]);
        int firstAltitude = Integer.parseInt(firstStrings[14]);
        double height = Double.parseDouble(firstStrings[3]);
        int azimuth = Integer.parseInt(firstStrings[4]);
        int electricalDowntilt = Integer.parseInt(firstStrings[5]);
        int mechanicalDowntilt = Integer.parseInt(firstStrings[6]);

        Writer writer = new FileWriter("misc/test.html");
        writer.write("<!DOCTYPE html>\n");
        writer.write("<html>\n");
        writer.write("<head>\n");
        writer.write("    <meta charset=\"utf-8\">\n");
        writer.write("    <title>ECharts</title>\n");
        writer.write("    <script src=\"https://cdn.staticfile.org/jquery/1.10.2/jquery.min.js\"></script>\n");
        writer.write("    <script src=\"https://cdn.bootcss.com/echarts/4.3.0-rc.2/echarts.js\"></script>\n");
        writer.write("    <script src=\"https://www.echartsjs.com/zh/dist/echarts-gl.min.js\"></script>\n");
        writer.write("</head>\n");
        writer.write("<body>\n");
        writer.write("    <div id=\"main\" style=\"width: 1600px;height:800px;\"></div>\n");
        writer.write("    <script type=\"text/javascript\">\n");
        writer.write("        // 基于准备好的dom，初始化echarts实例\n");
        writer.write("        var myChart = echarts.init(document.getElementById('main'));\n");
        writer.write("\n");
        writer.write("        myChart.setOption(option = {\n");
        writer.write("            visualMap: {\n");
        writer.write("                show: false,\n");
        writer.write("                min: -130,\n");
        writer.write("                max: -60,\n");
        writer.write("                dimension: 3,\n");
        writer.write("                inRange: {\n");
        writer.write("                    symbolSize: [1, 5],\n");
        writer.write("                    color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'],\n");
        writer.write("                    colorAlpha: [0.2, 1]\n");
        writer.write("                }\n");
        writer.write("            },\n");
        writer.write("            xAxis3D: {\n");
        writer.write("                scale: true,\n");
        writer.write("                type: 'value'\n");
        writer.write("            },\n");
        writer.write("            yAxis3D: {\n");
        writer.write("                scale: true,\n");
        writer.write("                type: 'value'\n");
        writer.write("            },\n");
        writer.write("            zAxis3D: {\n");
        writer.write("                scale: true,\n");
        writer.write("                type: 'value'\n");
        writer.write("            },\n");
        writer.write("            grid3D: {\n");
        writer.write("                axisLine: {\n");
        writer.write("                    lineStyle: { color: '#000' }\n");
        writer.write("                },\n");
        writer.write("                axisPointer: {\n");
        writer.write("                    lineStyle: { color: '#000' }\n");
        writer.write("                },\n");
        writer.write("                viewControl: {\n");
        writer.write("                    projection: 'orthographic'\n");
        writer.write("                }\n");
        writer.write("            },\n");
        writer.write("            series: [{\n");
        writer.write("                type: 'line3D',\n");
        writer.write("                data: ");
        // begin
        writer.write("[");
        writer.write("[0,0," + (firstAltitude + height) + "]");
        DecimalFormat decimalFormat = new DecimalFormat("0.000");
        for (int i = 1; i < 200; i += 1) {
            writer.write(",[" + decimalFormat.format(i * Math.sin(Math.toRadians(azimuth))) + "," + decimalFormat.format(i * Math.cos(Math.toRadians(azimuth))) + ","
                    + decimalFormat.format(firstAltitude + height - Math.tan(Math.toRadians(electricalDowntilt + mechanicalDowntilt)) * i) + "]");
        }
        writer.write("]");
        // end
        writer.write("\n            }, {\n");
        writer.write("                type: 'scatter3D',\n");
        writer.write("                data: ");
        // begin
        writer.write("[");
        int firstX = Integer.parseInt(firstStrings[12]);
        int firstY = Integer.parseInt(firstStrings[13]);
        double firstRSRP = Double.parseDouble(firstStrings[17]);
        //double firstRSRP = -100;
        writer.write("[" + (firstX - firstCellX) + "," + (firstY - firstCellY) + "," + firstAltitude + "," + firstRSRP + "]");
        while (true) {
            String string = bufferedReader.readLine();
            if (string == null) {
                break;
            }
            String[] strings = string.split(",");
            double CellX = Double.parseDouble(strings[1]);
            double CellY = Double.parseDouble(strings[2]);
            int X = Integer.parseInt(strings[12]);
            int Y = Integer.parseInt(strings[13]);
            int altitude = Integer.parseInt(strings[14]);
            double RSRP = Double.parseDouble(strings[17]);
            //double RSRP = -100;
            writer.write(",[" + (X - CellX) + "," + (Y- CellY) + "," + altitude + "," + RSRP + "]");
        }
        writer.write("]");
        // end
        writer.write("\n            }]\n");
        writer.write("        });\n");
        writer.write("    </script>\n");
        writer.write("</body>\n");
        writer.write("</html>");
        writer.flush();
    }
}
