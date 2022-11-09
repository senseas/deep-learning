package com.deep.framework.ast;

import com.deep.framework.ast.declaration.ClassOrInterfaceDeclaration;
import com.deep.framework.ast.declaration.MethodDeclaration;
import com.deep.framework.ast.expression.ArrayExpression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import com.deep.framework.ast.lexer.BlockLexer;
import com.deep.framework.ast.lexer.Lexer;
import com.deep.framework.ast.lexer.StringLexer;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import com.deep.framework.ast.statement.Statement;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

import static com.deep.framework.ast.lexer.TokenType.*;
import static javax.lang.model.SourceVersion.isIdentifier;

public class Parser {
    Lexer lexer = new BlockLexer(null);
    List<Object> list = new ArrayList<>();

    public void add(String a) {
        if (a.isEmpty()) return;

        TokenType type = TokenType.getType(a);
        if (Objects.nonNull(type)) {
            list.add(type);
        } else {
            list.add(a);
        }
    }

    public void parser(String strFile) {
        Stream<String> stream = FileUtil.readFile(strFile);
        stream.reduce((a, b) -> {
            if (lexer.getType().equals("string")) {
                if (a.concat(b).endsWith("\\\"")) return a.concat(b);
                if (!b.equals(StringLexer.end)) return a.concat(b);
                lexer = lexer.getLexer();
                add(a.concat(b));
                return "";
            } else if (b.equals(StringLexer.start)) {
                lexer = new StringLexer(lexer);
                add(a);
                return a.concat(b);
            } else if (Character.isWhitespace(b.charAt(0))) {
                add(a);
                return "";
            } else if (!a.isEmpty() && isIdentifier(a.concat(b))) {
                return a.concat(b);
            } else if (TokenType.startsWith(a.concat(b))) {
                return a.concat(b);
            } else {
                add(a);
                return b;
            }
        });

        CompilationUnit compilationUnit = new CompilationUnit();
        parserBlockStatement(compilationUnit);
        parserStatement(compilationUnit);
        reduce(compilationUnit);
    }

    private void parserBlockStatement(CompilationUnit compilationUnit) {
        Node node = new Node(compilationUnit);
        compilationUnit.getChildrens().add(node);
        for (Object o : list) {
            if (o.equals(TokenType.RBRACE)) {
                node = node.getPrarent();
            } else if (o.equals(TokenType.LBRACE)) {
                Node child = new BlockStatement(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(TokenType.RPAREN)) {
                node = node.getPrarent();
            } else if (o.equals(TokenType.LPAREN)) {
                Node child = new ParametersExpression(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(TokenType.RBRACK)) {
                node = node.getPrarent();
            } else if (o.equals(TokenType.LBRACK)) {
                Node child = new ArrayExpression(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o instanceof String) {
                Name child = new Name((String) o);
                node.getChildrens().add(child);
            } else {
                node.getChildrens().add(o);
            }
        }
    }

    public void parserStatement(Node node) {
        List<Object> list = new ArrayList<>();
        Statement a = new Statement(node);
        for (Object n : node.getChildrens()) {
            if (n instanceof BlockStatement) {
                list.add(a);
                list.add(n);
                a = new Statement(node);
                parserStatement((Node) n);
            } else if (n.equals(TokenType.SEMI)) {
                list.add(a);
                a = new Statement(node);
            } else if (n instanceof Node) {
                parserStatement((Node) n);
                a.getChildrens().add(n);
            } else {
                a.getChildrens().add(n);
            }
        }
        if (list.isEmpty()) return;
        node.setChildrens(list);
    }

    public void reduce(Node node) {
        parserClass(node);
        for (Object n : node.getChildrens()) {
            if (n instanceof Node) {
                reduce((Node) n);
            }
        }
    }

    public void parserClass(Node node) {
        List.copyOf(node.getChildrens()).stream().reduce((m, n) -> {
            if (m.equals(CLASS)) {
                ClassOrInterfaceDeclaration classDeclare = new ClassOrInterfaceDeclaration(node.getPrarent());
                List<Object> prarentChildrens = node.getPrarent().getChildrens();
                int index = prarentChildrens.indexOf(node);
                prarentChildrens.set(index, classDeclare);
                List.copyOf(node.getChildrens()).stream().reduce((a, b) -> {
                    if (a.equals(CLASS)) {
                        classDeclare.setName((Name) b);
                        node.getChildrens().remove(a);
                    } else if (List.of(PUBLIC, PRIVATE, PROTECTED).contains(a)) {
                        classDeclare.setModifier((TokenType) a);
                        node.getChildrens().remove(a);
                    }
                    return b;
                });
                BlockStatement body = (BlockStatement) prarentChildrens.get(index + 1);
                classDeclare.setBody(body);
                prarentChildrens.remove(body);
                classDeclare.setChildrens(node.getChildrens());
                classDeclare.getChildrens().add(body);
                parserMethod(body);
            }
            return n;
        });
    }

    public void parserMethod(Node node) {
        if (node instanceof MethodDeclaration) return;
        List.copyOf(node.getChildrens()).stream().map(a -> (Node) a).reduce((m, n) -> {
            if (m.getChildrens().size() > 2) {
                Object c = m.getChildrens().get(m.getChildrens().size() - 2);
                Object d = m.getChildrens().get(m.getChildrens().size() - 1);
                if (c instanceof Name && d instanceof ParametersExpression && n instanceof BlockStatement) {
                    MethodDeclaration methodDeclare = new MethodDeclaration(node.getPrarent());
                    methodDeclare.setName((Name) c);
                    methodDeclare.setParameters((ParametersExpression) d);
                    int index = node.getChildrens().indexOf(m);
                    node.getChildrens().set(index, methodDeclare);
                    node.getChildrens().remove(n);
                    m.getChildrens().add(n);
                    methodDeclare.setBody((BlockStatement) n);
                }
            }
            return n;
        });
    }
}