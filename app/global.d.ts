// Allow importing global CSS and CSS modules in TypeScript
// This file makes TS accept imports like `import './globals.css'`
// and `import styles from './Component.module.css'`.

declare module '*.module.css' {
  const classes: { [key: string]: string };
  export default classes;
}

declare module '*.module.scss' {
  const classes: { [key: string]: string };
  export default classes;
}

declare module '*.css';
declare module '*.scss';
declare module '*.sass';
declare module '*.less';
declare module '*.styl';
