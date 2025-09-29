// signage.js - sign payload with private RSA key
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const PRIVATE_PEM = path.join(__dirname, '..', 'certs', 'private.pem');
const PUBLIC_PEM = path.join(__dirname, '..', 'certs', 'public.pem');

function signPayload(payload) {
  const data = JSON.stringify(payload, Object.keys(payload).sort(), 2);
  const sign = crypto.createSign('SHA256');
  sign.update(data);
  sign.end();
  const privateKey = fs.readFileSync(PRIVATE_PEM, 'utf8');
  const signature = sign.sign(privateKey, 'base64');
  return { payload, signature, signed_at: new Date().toISOString() };
}

function verifySignature(signedPackage) {
  const publicKey = fs.readFileSync(PUBLIC_PEM, 'utf8');
  const verify = crypto.createVerify('SHA256');
  const data = JSON.stringify(signedPackage.payload, Object.keys(signedPackage.payload).sort(), 2);
  verify.update(data);
  verify.end();
  return verify.verify(publicKey, signedPackage.signature, 'base64');
}

module.exports = { signPayload, verifySignature };