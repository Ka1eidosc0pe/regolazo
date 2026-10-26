import "./style.css";

const app = document.querySelector<HTMLDivElement>("#app");
if (app) {
  app.innerHTML = `<h1 data-testid="title">ReGolazo</h1>`;
}
