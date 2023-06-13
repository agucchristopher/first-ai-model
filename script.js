const tf = require("@tensorflow/tfjs");

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [2] }));

model.compile({ loss: "meanSquaredErro", optimizer: "sgd" });

const input = tf.tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]);
const output = tf.tensor([[0], [1], [1], [2]]);

model.fit(input, output, { epochs: 10 }).then(() => {
  const testData = tf.tensor([
    [0, 1],
    [1, 1],
    [1, 0],
  ]);
  const predictions = model.predict(testData);
  predictions.print();
});
