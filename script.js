const tf = require("@tensorflow/tfjs");

try {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [2] }));

  model.compile({ loss: "meanSquaredErro", optimizer: "sgd" });

  const input = tf.tensor([
    [1, 1],
    [0, 0],
  ]);
  const output = tf.tensor([[0], [1]]);

  model.fit(input, output, { epochs: 10 }).then(() => {
    const testData = tf.tensor([
      [1, 1],
      [0, 0],
    ]);
    const predictions = model.predict(testData);
    predictions.print();
  });
} catch (error) {
  console.log(error.message);
}
